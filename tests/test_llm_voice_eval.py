"""Unit tests for LLM evaluation metrics and Voice STT WER evaluation."""

from __future__ import annotations

from src.evaluation.llm_eval import evaluate_sql_pair, run_llm_evaluation
from src.evaluation.voice_eval import compute_wer_metrics


def test_evaluate_sql_pair_valid_and_exact():
    gen_sql = "SELECT customer_id FROM customers WHERE country = 'India';"
    gold_sql = "SELECT customer_id FROM customers WHERE country = 'India';"
    res = evaluate_sql_pair(gen_sql, gold_sql, execution_match=True, known_tables={"customers"})

    assert res["valid_sql"] is True
    assert res["exact_match"] is True
    assert res["execution_match"] is True
    assert res["schema_compliant"] is True


def test_evaluate_sql_pair_schema_hallucination():
    gen_sql = "SELECT * FROM non_existent_table;"
    gold_sql = "SELECT * FROM customers;"
    res = evaluate_sql_pair(gen_sql, gold_sql, execution_match=False, known_tables={"customers"})

    assert res["valid_sql"] is True
    assert res["exact_match"] is False
    assert res["schema_compliant"] is False


def test_run_llm_evaluation_summary():
    dataset = [
        {"generated_sql": "SELECT 1;", "gold_sql": "SELECT 1;", "ok": True},
        {"generated_sql": "INVALID SQL...", "gold_sql": "SELECT 1;", "ok": False},
    ]
    metrics = run_llm_evaluation(dataset, known_tables={"customers"})
    summary = metrics.to_dict()

    assert summary["total_examples"] == 2
    assert summary["metrics"]["execution_accuracy_ex_percent"] == 50.0
    assert summary["metrics"]["valid_sql_rate_percent"] == 50.0


def test_compute_wer_metrics():
    refs = ["show top 5 customers from india", "list all pending orders"]
    hyps = ["show top 5 customers from india", "list all pending order"]
    metrics = compute_wer_metrics(refs, hyps)

    assert "word_error_rate_wer" in metrics
    assert "match_error_rate_mer" in metrics
    assert metrics["sample_count"] == 2
    assert metrics["word_error_rate_wer"] >= 0.0
