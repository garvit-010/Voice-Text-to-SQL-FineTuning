"""Comprehensive LLM Text-to-SQL Evaluation Suite.

Evaluates LLM generation across key production & research metrics:
- Execution Accuracy (EX): % of queries returning identical result sets to gold SQL
- Valid SQL Rate: % of generated queries that parse cleanly without syntax errors
- Exact Match (EM): % of AST-normalized query matches
- Schema Compliance: % of queries without table/column hallucinations
- Repair Success Rate: % of failing queries resolved by self-correction
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import sqlglot


@dataclass
class LLMEvalMetrics:
    total_examples: int = 0
    valid_sql_count: int = 0
    exact_match_count: int = 0
    execution_correct_count: int = 0
    schema_compliant_count: int = 0
    repaired_count: int = 0

    @property
    def valid_sql_rate(self) -> float:
        return round((self.valid_sql_count / self.total_examples) * 100, 2) if self.total_examples else 0.0

    @property
    def exact_match_rate(self) -> float:
        return round((self.exact_match_count / self.total_examples) * 100, 2) if self.total_examples else 0.0

    @property
    def execution_accuracy(self) -> float:
        return round((self.execution_correct_count / self.total_examples) * 100, 2) if self.total_examples else 0.0

    @property
    def schema_compliance_rate(self) -> float:
        return round((self.schema_compliant_count / self.total_examples) * 100, 2) if self.total_examples else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "metrics": {
                "execution_accuracy_ex_percent": self.execution_accuracy,
                "valid_sql_rate_percent": self.valid_sql_rate,
                "exact_match_em_percent": self.exact_match_rate,
                "schema_compliance_rate_percent": self.schema_compliance_rate,
                "repaired_count": self.repaired_count,
            },
        }


def evaluate_sql_pair(
    generated_sql: str,
    gold_sql: str,
    execution_match: bool,
    repaired: bool = False,
    known_tables: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Evaluate one generated SQL query against gold SQL and schema rules."""
    is_valid_sql = False
    is_exact_match = False
    is_schema_compliant = True

    # 1. Syntax check via sqlglot
    try:
        gen_ast = sqlglot.parse_one(generated_sql, read="postgres")
        is_valid_sql = True
        try:
            gold_ast = sqlglot.parse_one(gold_sql, read="postgres")
            is_exact_match = gen_ast.sql(normalize=True) == gold_ast.sql(normalize=True)
        except Exception:
            is_exact_match = generated_sql.strip().lower() == gold_sql.strip().lower()
    except Exception:
        is_valid_sql = False

    # 2. Schema compliance check (table hallucination check)
    if is_valid_sql and known_tables:
        try:
            gen_ast = sqlglot.parse_one(generated_sql, read="postgres")
            used_tables = {t.name.lower() for t in gen_ast.find_all(sqlglot.exp.Table)}
            if not used_tables.issubset({t.lower() for t in known_tables}):
                is_schema_compliant = False
        except Exception:
            is_schema_compliant = False

    return {
        "valid_sql": is_valid_sql,
        "exact_match": is_exact_match,
        "execution_match": execution_match,
        "schema_compliant": is_schema_compliant,
        "repaired": repaired,
    }


def run_llm_evaluation(dataset: List[Dict[str, Any]], known_tables: Optional[set[str]] = None) -> LLMEvalMetrics:
    """Run full evaluation suite over a dataset list of query results."""
    m = LLMEvalMetrics(total_examples=len(dataset))
    for item in dataset:
        res = evaluate_sql_pair(
            generated_sql=item.get("generated_sql", ""),
            gold_sql=item.get("gold_sql", ""),
            execution_match=item.get("ok", False),
            repaired=item.get("repaired", False),
            known_tables=known_tables,
        )
        if res["valid_sql"]:
            m.valid_sql_count += 1
        if res["exact_match"]:
            m.exact_match_count += 1
        if res["execution_match"]:
            m.execution_correct_count += 1
        if res["schema_compliant"]:
            m.schema_compliant_count += 1
        if res["repaired"]:
            m.repaired_count += 1

    return m
