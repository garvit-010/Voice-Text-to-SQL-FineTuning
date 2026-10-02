"""FastAPI service exposing the Text-to-SQL pipeline.

Phase 12.

    POST /query    question in, rows plus the SQL that produced them out
    GET  /schema   what the model is told about the database
    GET  /health   database and model readiness
    GET  /docs     interactive OpenAPI (FastAPI built-in)

Run it (from the project root, so ``src`` is importable):

    env\\Scripts\\python.exe -m uvicorn src.api.main:app --reload

Design notes worth stating, because they are the difference between a demo and
a service:

*The model and the schema are built once, at start-up.* Loading them per
request would add seconds to every call and re-read a schema that cannot change
without a restart anyway.

*The connection pool is read-only and time-limited.* Every connection is
configured with ``default_transaction_read_only`` and a statement timeout
before it is handed out, so a query that escapes validation still cannot write
and still cannot hang the server.

*A bad question is not a server error.* Generated SQL that fails to parse or
fails to execute returns HTTP 200 with ``ok: false`` and the reason. Reserving
5xx for genuine server faults is what makes the error rate meaningful.

*Cross-origin calls are allowed from Hugging Face Spaces only.* The public
demo is a static Space that calls this API from the browser, so the browser
needs a CORS grant. It is scoped to ``*.hf.space`` plus whatever
``CORS_ORIGINS`` lists, never ``*``: the API is read-only and unauthenticated,
so the risk is not data exposure but someone else's page spending this
deployment's inference quota.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.observability import metrics, log_structured_event, trace_llm_query, trace_voice_transcription
from src.api.backends import build_model
from src.api.schemas import (
    AttemptInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
    Timings,
    VoiceResponse,
)
from src.api.service import TextToSQLService
from src.api.backends import resolve_prompt
from src.sql.config import ConfigError, app_config
from src.sql.executor import DEFAULT_STATEMENT_TIMEOUT_MS

log = logging.getLogger("text2sql.api")

STATEMENT_TIMEOUT_MS = int(
    os.getenv("SQL_STATEMENT_TIMEOUT_MS", str(DEFAULT_STATEMENT_TIMEOUT_MS)))


def _configure_connection(conn) -> None:
    """Make every pooled connection read-only and time-limited.

    Applied on checkout rather than once at creation: a pooled connection is
    reused across requests, and a session setting reset by anything at all
    would silently remove the guarantee.
    """
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        cur.execute("SET default_transaction_read_only = on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pool, the model and the schema context once."""
    from psycopg_pool import ConnectionPool

    app.state.ready = False
    app.state.startup_error = None
    app.state.pool = None
    app.state.service = None

    try:
        cfg = app_config()
    except ConfigError as exc:
        # Start anyway so /health can explain what is wrong. Refusing to boot
        # makes a misconfigured deployment look like a crash loop.
        app.state.startup_error = str(exc)
        log.error("configuration error: %s", exc)
        yield
        return

    try:
        pool = ConnectionPool(
            cfg.conninfo(),
            min_size=int(os.getenv("DB_POOL_MIN", "1")),
            max_size=int(os.getenv("DB_POOL_MAX", "4")),
            configure=_configure_connection,
            open=True,
            timeout=10.0,
        )
        app.state.pool = pool

        model = build_model()
        with pool.connection() as conn:
            app.state.service = TextToSQLService.from_connection(model, conn)
        app.state.ready = True
        log.info("ready: model=%s schema=%s",
                 model.model_id, app.state.service.schema_fingerprint)
    except Exception as exc:  # noqa: BLE001 - reported through /health
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        log.exception("startup failed")

    yield

    if app.state.pool is not None:
        app.state.pool.close()





app = FastAPI(
    title="Enterprise Text-to-SQL",
    version="1.0.0",
    summary="Natural-language questions answered as executed PostgreSQL.",
    description=(
        "Converts business questions into PostgreSQL, validates the SQL, runs "
        "it against a read-only session and returns the rows. On a detectable "
        "failure it feeds the database error back to the model and retries "
        "once.\n\n"
        "Accuracy depends on the configured backend: the base model scores "
        "10.82 % strict execution accuracy on the project benchmark, the "
        "fine-tuned adapter 50.99 %."
    ),
    lifespan=lifespan,
)

# Browsers enforce CORS, servers grant it. The static Space lives on a
# *.hf.space origin; extra origins (a local build, a custom domain) come from
# the environment as a comma-separated list. No credentials are involved, so
# there is nothing a cross-site request could act on behalf of.
_extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_extra_origins,
    allow_origin_regex=r"https://[a-z0-9-]+\.(static\.)?hf\.space",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


def _service(request: Request) -> TextToSQLService:
    """The live service, or a 503 explaining why there is not one."""
    service = getattr(request.app.state, "service", None)
    if service is None or not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.startup_error or "service is starting",
        )
    return service


@app.get("/metrics", tags=["observability"])
def get_metrics() -> dict:
    """Return runtime operational performance, latency, and query metrics."""
    return metrics.get_summary()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Readiness, including whether the database actually answers.

    Deliberately executes ``SELECT 1`` rather than trusting that a pool exists.
    A pool object with no reachable database is exactly the failure a health
    check is meant to catch.
    """
    state = request.app.state
    db_ok = False
    detail = state.startup_error

    pool = getattr(state, "pool", None)
    if pool is not None:
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            db_ok = True
        except Exception as exc:  # noqa: BLE001
            detail = f"database unreachable: {type(exc).__name__}"

    service = getattr(state, "service", None)
    return HealthResponse(
        status="ok" if (db_ok and service is not None) else "degraded",
        database=db_ok,
        model=service is not None,
        model_id=service.model.model_id if service else None,
        schema_fingerprint=service.schema_fingerprint if service else None,
        prompt_fingerprint=(service.model.prompt.prompt_fingerprint()
                            if service is not None else None),
        detail=detail,
    )


@app.get("/schema", response_model=SchemaResponse, tags=["introspection"])
def schema(request: Request) -> SchemaResponse:
    """Exactly what the model is shown. Useful when a result looks wrong."""
    service = _service(request)
    return SchemaResponse(
        fingerprint=service.schema_fingerprint,
        characters=len(service.schema_text),
        tables=sorted(service.schema_info.tables),
        schema_text=service.schema_text,
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer one question.

    Returns 200 even when the generated SQL fails; ``ok`` carries that. 5xx is
    reserved for the service itself being broken.
    """
    service = _service(request)
    pool = request.app.state.pool

    started = time.perf_counter()
    try:
        with pool.connection() as conn:
            result = service.answer(
                body.question, conn,
                max_rows=body.max_rows,
                repair=body.repair,
                include_rows=body.include_rows,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failure")
        raise HTTPException(
            status_code=503,
            detail=f"pipeline unavailable: {type(exc).__name__}",
        ) from exc

    # Observability recording & tracing
    metrics.record_query(
        ok=result.ok,
        repaired=result.repaired,
        generation_ms=result.generation_ms,
        execution_ms=result.execution_ms,
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        error_stage=result.error_stage,
    )
    trace_llm_query(
        question=body.question,
        model_name=service.model.model_id,
        sql=result.sql,
        ok=result.ok,
        repaired=result.repaired,
        row_count=result.row_count,
        generation_ms=result.generation_ms,
        execution_ms=result.execution_ms,
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        error_stage=result.error_stage,
    )
    log_structured_event("query_processed", {
        "question": body.question,
        "ok": result.ok,
        "repaired": result.repaired,
        "sql": result.sql,
        "row_count": result.row_count,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "error_stage": result.error_stage,
    })

    return QueryResponse(
        question=body.question,
        ok=result.ok,
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
        repaired=result.repaired,
        attempts=[AttemptInfo(**a) for a in result.attempts],
        error=result.error,
        error_stage=result.error_stage,
        timings=Timings(
            generation_ms=result.generation_ms,
            repair_generation_ms=result.repair_generation_ms,
            execution_ms=result.execution_ms,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        ),
        model=service.model.model_id,
    )


@app.post("/voice", response_model=VoiceResponse, tags=["voice"])
async def voice(audio: UploadFile = File(...)) -> VoiceResponse:
    """Transcribe an audio file to text using OpenAI Whisper.

    Accepts any audio format Whisper supports (webm, mp3, wav, m4a, ogg).
    Returns the transcript, which the caller can then send to /query.

    Requires either:
    - OPENAI_API_KEY in .env, or
    - HF_TOKEN with inference provider access (uses HF's Whisper endpoint)
    """
    import httpx

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25 MB Whisper limit
        raise HTTPException(status_code=413, detail="Audio file too large (max 25 MB)")

    gemini_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GEMINI_API", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    filename = audio.filename or "audio.webm"
    transcript = ""

    if gemini_key:
        try:
            import asyncio
            from google import genai
            from google.genai import types

            content_type = (audio.content_type or "").lower()
            if "wav" in content_type:
                mime_type = "audio/wav"
            elif "mp3" in content_type or "mpeg" in content_type:
                mime_type = "audio/mp3"
            elif "ogg" in content_type:
                mime_type = "audio/ogg"
            elif "m4a" in content_type or "mp4" in content_type:
                mime_type = "audio/mp4"
            else:
                mime_type = "audio/webm"

            def _transcribe():
                client = genai.Client(api_key=gemini_key)
                models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite"]
                last_err = None
                for model_name in models_to_try:
                    try:
                        res = client.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                                "Transcribe this spoken audio accurately into text. Output ONLY the raw transcript text, without any quotes, intro/outro, or commentary."
                            ]
                        )
                        return (res.text or "").strip()
                    except Exception as e:
                        last_err = e
                        continue
                if last_err:
                    raise last_err
                return ""

            transcript = await asyncio.get_event_loop().run_in_executor(None, _transcribe)
        except Exception as exc:
            log.exception("Gemini audio transcription error")
            raise HTTPException(
                status_code=502,
                detail=f"Gemini Speech error: {type(exc).__name__}: {str(exc)[:200]}",
            ) from exc

    elif openai_key:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_key}"},
                files={"file": (filename, audio_bytes, audio.content_type or "audio/webm")},
                data={"model": "whisper-1"},
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=502,
                detail=f"Whisper API error: {resp.status_code} {resp.text[:200]}")
        transcript = resp.json().get("text", "").strip()

    elif hf_token:
        # Use HuggingFace InferenceClient SDK — avoids raw DNS issues with
        # api-inference.huggingface.co that some ISPs / networks block.
        try:
            import asyncio
            from huggingface_hub import InferenceClient

            hf_client = InferenceClient(token=hf_token)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: hf_client.automatic_speech_recognition(
                    audio_bytes,
                    model="openai/whisper-large-v3",
                ),
            )
            transcript = (result.text or "").strip()
        except Exception as exc:
            log.exception("HF Whisper SDK error")
            raise HTTPException(
                status_code=502,
                detail=f"HF Whisper error: {type(exc).__name__}: {str(exc)[:200]}",
            ) from exc

    else:
        raise HTTPException(
            status_code=503,
            detail="No speech-to-text backend configured. "
                   "Set GEMINI_API_KEY, OPENAI_API_KEY, or HF_TOKEN in .env",
        )

    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio — try again")

    metrics.record_voice()
    trace_voice_transcription(transcript=transcript, word_count=len(transcript.split()))
    log_structured_event("voice_transcribed", {
        "words": len(transcript.split()),
        "length_chars": len(transcript),
    })
    return VoiceResponse(transcript=transcript, words=len(transcript.split()))


async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback, a file path or a connection string to a caller."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "type": type(exc).__name__},
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def root() -> Response:
    """The demo UI, or a JSON descriptor if the static file is absent.

    Falling back rather than 404-ing matters for the container: the API is
    useful headless, and a missing `static/` should not take the service down.
    """
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({
        "service": "Enterprise Text-to-SQL",
        "docs": "/docs",
        "health": "/health",
        "prompt_version": resolve_prompt().PROMPT_VERSION,
    })
