"""Tests for the observability metrics collector and endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api.main import app
from src.api.observability import MetricsCollector


def test_metrics_collector_unit():
    collector = MetricsCollector()
    collector.record_query(ok=True, repaired=False, generation_ms=150.0, execution_ms=20.0, total_ms=170.0)
    collector.record_query(ok=False, repaired=True, generation_ms=200.0, execution_ms=0.0, total_ms=200.0, error_stage="execution")
    collector.record_voice()

    summary = collector.get_summary()
    assert summary["queries"]["total"] == 2
    assert summary["queries"]["successful"] == 1
    assert summary["queries"]["failed"] == 1
    assert summary["queries"]["repaired"] == 1
    assert summary["voice"]["transcriptions_total"] == 1
    assert summary["error_stages"]["execution"] == 1


def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "queries" in data
    assert "latency_avg_ms" in data
    assert "uptime_seconds" in data
