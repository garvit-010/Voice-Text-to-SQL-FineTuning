# Evaluation & Experiment Reports Index

This directory consolidates all official benchmark reports, ablation studies, and failure analyses across **v1** and **v2** iterations of the **Enterprise Voice & Text-to-SQL Fine-Tuning** pipeline.

All evaluations were executed on a held-out, sealed test set of **453 queries** (SHA-256 fingerprint: `ec7ddcae4f9d90d4`), evaluated using live PostgreSQL execution comparisons against ground-truth queries.

---

## Benchmark Highlights & Executive Summary

### v2 Progression (Current Production Standard)

| Milestone / Configuration | Strict Execution Accuracy (EX) | Syntax Validity | Avg Latency | Notes |
|:---|:---:|:---:|:---:|:---|
| **Base Model (Qwen3-8B) + Prompt v2** | **43.71%** (198/453) | 91.83% | 1.84s | Zero-shot baseline with schema context + business glossary |
| **Fine-Tuned Adapter (QLoRA) + Prompt v2** | **68.43%** (310/453) | 98.45% | 1.42s | +24.72 pp gain from domain SFT on 2,133 enterprise samples |
| **Fine-Tuned + Self-Correction Repair (v2)** | **70.86%** (321/453) | 99.78% | 1.95s | **Production Headline Result** (+27.15 pp over base) |

### v2 Accuracy Breakdown by Query Difficulty

| Query Complexity Tier | Base Model (v2) | Fine-Tuned (v2) | Fine-Tuned + Repair (v2) | Relative Lift |
|:---|:---:|:---:|:---:|:---:|
| **Easy** (Single table, basic filters/aggregates) | 88.2% | 98.4% | **100.0%** | +11.8 pp |
| **Medium** (2-3 table joins, GROUP BY, subqueries) | 41.5% | 71.9% | **75.4%** | +33.9 pp |
| **Hard** (Multi-join, complex CTEs, window functions) | 16.7% | 54.2% | **62.5%** | +45.8 pp |

---

## Document Catalog

### Version 2 (Production Milestone)
1. **[v2_ablation_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_ablation_report.md)**:
   Full ablation comparative study evaluating prompt v2 changes, base model vs fine-tuned adapter, and the effect of the iterative self-correction repair loop.
2. **[v2_baseline_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_baseline_report.md)**:
   Baseline evaluation of untuned Qwen3-8B utilizing enhanced Prompt v2 (business glossary, deterministic `DATA_AS_OF` date anchor, NULL conventions).
3. **[v2_finetuned_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_finetuned_report.md)**:
   Evaluation of the QLoRA adapter (`garvit-010/qwen3-8b-text2sql-qlora-v3`) with single-pass generation.
4. **[v2_repair_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_repair_report.md)**:
   Detailed analysis of the self-correction repair loop. Out of 25 failed generation attempts, 17 were successfully repaired to execute (68% repair execution rate), and 11 returned exact gold result rows (44% correctness rate).
5. **[v2_failure_analysis.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_failure_analysis.md)**:
   Taxonomy of remaining failure modes in single-pass fine-tuned inference (categorized into schema mismatch, join path ambiguity, and complex date arithmetic).
6. **[v2_failure_analysis_repair_rescored.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v2_failure_analysis_repair_rescored.md)**:
   Post-repair residual error analysis explaining which persistent error classes resisted self-correction.

---

### Version 1 (Historical Baseline & Exploration)
1. **[v1_ablation_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_ablation_report.md)**:
   Initial ablation across 5 configurations, identifying that keyword-based schema retrieval degraded performance on small schemas (context fits easily, retrieval introduced miss-rates).
2. **[v1_baseline_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_baseline_report.md)**:
   Initial zero-shot baseline on Prompt v1 (10.82% strict EX).
3. **[v1_finetuned_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_finetuned_report.md)**:
   Initial QLoRA fine-tuning result (50.99% strict EX).
4. **[v1_finetuned_retrieval_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_finetuned_retrieval_report.md)**:
   Empirical ablation demonstrating why dynamic keyword schema pruning hurt accuracy (dropped from 50.99% to 41.72%).
5. **[v1_repair_report.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_repair_report.md)**:
   Initial prototype of the automated repair loop (52.10% strict EX).
6. **[v1_failure_analysis.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/v1_failure_analysis.md)**:
   Early error breakdown driving the design of Prompt v2 (highlighting ambiguous column projections and lack of date grounding).
