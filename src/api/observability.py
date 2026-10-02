"""Production Observability, Langfuse Tracing & Metrics for Text-to-SQL API.

Provides:
- In-memory operational metrics collection for /metrics endpoint
- Structured JSON logging for cloud log aggregation
- Full Langfuse LLM tracing & telemetry (spans, generations, execution scoring)
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


# --- Langfuse LLM Tracing ---

_langfuse_client = None
_langfuse_init_attempted = False
_langfuse_lock = threading.Lock()


def get_langfuse_client() -> Optional[Any]:
    """Lazy initialize and return the global Langfuse client instance."""
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
                except ImportError:
                    log.warning("langfuse SDK not installed; tracing disabled")
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
    """Record a complete LLM text-to-SQL trace with Langfuse."""
    lf = get_langfuse_client()
    if not lf:
        return

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
    except Exception as exc:
        log.warning("Failed to send trace to Langfuse: %s", exc)


def trace_voice_transcription(transcript: str, word_count: int) -> None:
    """Record a voice transcription trace with Langfuse."""
    lf = get_langfuse_client()
    if not lf:
        return

    try:
        lf.trace(
            name="voice-transcription",
            input="audio_file",
            output={"transcript": transcript, "word_count": word_count},
        )
        lf.flush()
    except Exception as exc:
        log.warning("Failed to send voice trace to Langfuse: %s", exc)
