"""Production Observability, Langfuse & LangSmith Tracing & Metrics.

Provides:
- In-memory operational metrics collection for /metrics endpoint
- Structured JSON logging for cloud log aggregation
- Multi-provider LLM Telemetry (Langfuse & LangSmith RunTree tracing for HuggingFace / custom models)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import threading

log = logging.getLogger("text2sql.observability")


@dataclass
class QueryMetrics:
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    repaired_queries: int = 0
    voice_transcriptions: int = 0
    total_generation_ms: float = 0.0
    total_execution_ms: float = 0.0
    total_duration_ms: float = 0.0
    error_stages: Counter = field(default_factory=Counter)
    tables_queried: Counter = field(default_factory=Counter)


class MetricsCollector:
    """Thread-safe collector for API runtime performance & operational metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = QueryMetrics()
        self._start_time = time.time()

    def record_query(
        self,
        ok: bool,
        repaired: bool = False,
        generation_ms: float = 0.0,
        execution_ms: float = 0.0,
        total_ms: float = 0.0,
        error_stage: Optional[str] = None,
        tables: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            m = self._metrics
            m.total_queries += 1
            if ok:
                m.successful_queries += 1
            else:
                m.failed_queries += 1

            if repaired:
                m.repaired_queries += 1

            m.total_generation_ms += max(0.0, generation_ms)
            m.total_execution_ms += max(0.0, execution_ms)
            m.total_duration_ms += max(0.0, total_ms)

            if error_stage:
                m.error_stages[error_stage] += 1

            if tables:
                for t in tables:
                    m.tables_queried[t] += 1

    def record_voice(self) -> None:
        with self._lock:
            self._metrics.voice_transcriptions += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            m = self._metrics
            uptime_seconds = round(time.time() - self._start_time, 2)
            total = m.total_queries
            succ = m.successful_queries

            avg_gen = round(m.total_generation_ms / total, 2) if total > 0 else 0.0
            avg_exec = round(m.total_execution_ms / total, 2) if total > 0 else 0.0
            avg_total = round(m.total_duration_ms / total, 2) if total > 0 else 0.0
            success_rate = round((succ / total) * 100, 2) if total > 0 else 100.0

            return {
                "uptime_seconds": uptime_seconds,
                "queries": {
                    "total": total,
                    "successful": succ,
                    "failed": m.failed_queries,
                    "repaired": m.repaired_queries,
                    "success_rate_percent": success_rate,
                },
                "voice": {
                    "transcriptions_total": m.voice_transcriptions,
                },
                "latency_avg_ms": {
                    "generation": avg_gen,
                    "execution": avg_exec,
                    "total": avg_total,
                },
                "error_stages": dict(m.error_stages),
                "top_tables_queried": dict(m.tables_queried.most_common(10)),
            }


# Singleton instance shared across the service
metrics = MetricsCollector()


def log_structured_event(event_type: str, data: Dict[str, Any]) -> None:
    """Log structured JSON event for log aggregation (Render, CloudWatch, Datadog)."""
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        **data,
    }
    log.info("OBSERVABILITY_EVENT: %s", json.dumps(payload))


# --- Langfuse Tracing ---

_langfuse_client = None
_langfuse_init_attempted = False
_langfuse_lock = threading.Lock()


def get_langfuse_client() -> Optional[Any]:
    global _langfuse_client, _langfuse_init_attempted
    with _langfuse_lock:
        if not _langfuse_init_attempted:
            _langfuse_init_attempted = True
            pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
            sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
            if pk and sk:
                try:
                    from langfuse import Langfuse
                    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
                    _langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
                    log.info("Langfuse tracing initialized successfully (host=%s)", host)
                except Exception as e:
                    log.warning("Failed to initialize Langfuse client: %s", e)
    return _langfuse_client


def trace_llm_query(
    question: str,
    model_name: str,
    sql: str,
    ok: bool,
    repaired: bool = False,
    row_count: int = 0,
    generation_ms: float = 0.0,
    execution_ms: float = 0.0,
    total_ms: float = 0.0,
    error_stage: Optional[str] = None,
) -> None:
    """Record LLM text-to-SQL traces to active observability platforms (Langfuse & LangSmith)."""

    # 1. Langfuse Tracing
    lf = get_langfuse_client()
    if lf:
        try:
            trace = lf.trace(
                name="text-to-sql-query",
                input={"question": question},
                output={"sql": sql, "ok": ok, "row_count": row_count},
                metadata={
                    "model": model_name,
                    "repaired": repaired,
                    "error_stage": error_stage,
                    "generation_ms": generation_ms,
                    "execution_ms": execution_ms,
                    "total_ms": total_ms,
                },
            )
            trace.generation(
                name="sql-generation",
                model=model_name,
                input=question,
                output=sql,
                metadata={"repaired": repaired},
            )
            trace.score(
                name="execution_accuracy",
                value=1.0 if ok else 0.0,
                comment=f"Error stage: {error_stage}" if not ok else "Executed successfully",
            )
            lf.flush()
            log.info("Langfuse trace sent for question: %r", question[:30])
        except Exception as exc:
            log.warning("Failed to send trace to Langfuse: %s", exc)

    # 2. LangSmith Tracing via RunTree
    ls_api_key = (
        os.getenv("LANGSMITH_API_KEY", "").strip()
        or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )
    if ls_api_key:
        try:
            from langsmith import RunTree
            project_name = (
                os.getenv("LANGSMITH_PROJECT", "").strip()
                or os.getenv("LANGCHAIN_PROJECT", "voice-text-to-sql").strip()
            )
            # Ensure environment variables are visible to RunTree
            os.environ["LANGSMITH_API_KEY"] = ls_api_key
            os.environ["LANGCHAIN_API_KEY"] = ls_api_key

            rt = RunTree(
                name="text-to-sql-query",
                run_type="llm",
                inputs={"question": question},
                outputs={"sql": sql, "ok": ok, "row_count": row_count},
                project_name=project_name,
                extra={
                    "metadata": {
                        "model": model_name,
                        "repaired": repaired,
                        "error_stage": error_stage,
                        "generation_ms": generation_ms,
                        "execution_ms": execution_ms,
                        "total_ms": total_ms,
                        "execution_accuracy": 1.0 if ok else 0.0,
                    }
                },
            )
            rt.post()
            log.info("LangSmith RunTree trace posted successfully for question: %r", question[:30])
        except Exception as exc:
            log.warning("Failed to send trace to LangSmith RunTree: %s", exc)


def trace_voice_transcription(transcript: str, word_count: int) -> None:
    """Record voice transcription traces to active observability platforms."""

    # 1. Langfuse
    lf = get_langfuse_client()
    if lf:
        try:
            lf.trace(
                name="voice-transcription",
                input="audio_file",
                output={"transcript": transcript, "word_count": word_count},
            )
            lf.flush()
        except Exception as exc:
            log.warning("Failed to send voice trace to Langfuse: %s", exc)

    # 2. LangSmith
    ls_api_key = (
        os.getenv("LANGSMITH_API_KEY", "").strip()
        or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )
    if ls_api_key:
        try:
            from langsmith import RunTree
            project_name = (
                os.getenv("LANGSMITH_PROJECT", "").strip()
                or os.getenv("LANGCHAIN_PROJECT", "voice-text-to-sql").strip()
            )
            rt = RunTree(
                name="voice-transcription",
                run_type="tool",
                inputs={"audio": "uploaded_audio"},
                outputs={"transcript": transcript, "word_count": word_count},
                project_name=project_name,
            )
            rt.post()
        except Exception as exc:
            log.warning("Failed to send voice trace to LangSmith: %s", exc)
