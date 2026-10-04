# Complete Technical Interview Guide: Enterprise Voice & Text-to-SQL

This document is your complete playbook for explaining the **Enterprise Voice & Text-to-SQL System** during technical interviews (Machine Learning Engineer, LLM Engineer, MLOps, or Applied AI roles).

---

## 1. The 60-Second Elevator Pitch

> *"I engineered an end-to-end enterprise NLP system that translates natural language and voice queries into production-grade PostgreSQL SQL. The centerpiece is a QLoRA fine-tuned Qwen3-8B model that achieves **70.86% strict execution accuracy** on a frozen, held-out test set, jumping from a 43.71% zero-shot baseline and a 10.82% v1 base model.*
>
> *Unlike synthetic benchmark vanity metrics, every prediction is validated against a live 12-table enterprise PostgreSQL database by cryptographically comparing execution result sets against ground-truth rows. The system features an automated self-correction repair loop, strict AST-based safety guards, full LangSmith observability with sub-operation tracing, dual-path speech-to-text with Gemini Audio and Whisper, and is deployed via Docker on Render with automated CI/CD and cryptographic test set verification."*

---

## 2. High-Level Architecture & End-to-End Data Flow

```
                      +---------------------------------------+
                      |       User Interaction Channels       |
                      |  - Web UI / Text Query (`/query`)     |
                      |  - Audio / Voice Input (`/voice`)     |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |      Voice Transcription Layer        |
                      |  - Gemini 2.5 Flash Audio (Primary)   |
                      |  - Fallback: Whisper Large v3         |
                      |  - Phonetic Normalization Engine      |
                      +-------------------+-------------------+
                                          | (Clean Text Question)
                      +-------------------v-------------------+
                      |       Prompt & Context Assembly       |
                      |  - Deterministic Schema Context (DDL) |
                      |  - Business Glossary & Metric Rules   |
                      |  - Temporal Anchor (`DATA_AS_OF`)     |
                      |  - Prompt Fingerprint Verification    |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |       LLM Generation Engine           |
                      |  - Qwen3-8B + QLoRA Adapter (v3)      |
                      |  - Inference: HF Provider / Local     |
                      |  - Thinking Mode Suppressed (/no_think)|
                      +-------------------+-------------------+
                                          | (Raw SQL String)
                      +-------------------v-------------------+
                      |      Static AST & Safety Guardrails   |
                      |  - sqlglot AST Parsing                |
                      |  - Strict SELECT-Only Classification   |
                      |  - Schema Whitelist Column/Table Check|
                      +---------+-------------------+---------+
                                | Pass              | Fail
                                |                   |
                      +---------v---------+         |
                      | Live DB Execution |         |
                      | - Read-Only Conn  |         |
                      | - 30s Timeout     |         |
                      +---------+---------+         |
                                | Error             |
                                +---------> +-------v-------------------+
                                            |   Self-Correction Loop    |
                                            | - Error Trace Feedback    |
                                            | - 1-Turn Repair Inference |
                                            | - AST & Re-execution Check|
                                            +---------+-----------------+
                                                      |
                      +-------------------v-----------v-------+
                      |   Observability & Response Delivery   |
                      |  - LangSmith RunTree Spans            |
                      |  - Cryptographic Row Result Hashing   |
                      |  - Operational Latency Breakdown      |
                      |  - Return HTTP 200 + Structured JSON  |
                      +---------------------------------------+
```

---

## 3. Core Technical Decisions & Justifications

### Q1: Why fine-tune a model rather than prompting GPT-4o or Claude 3.5 Sonnet?
- **Enterprise Data Sovereignty & Cost**: Proprietary frontier models cost ~$5 to $15 per million tokens. Running hundreds of thousands of analytic queries through frontier models creates immense operational costs and vendor lock-in.
- **Domain & Schema Specialization**: Generalist models lack implicit knowledge of specific organizational conventions (e.g., in our schema, "revenue" strictly means `SUM(order_items.unit_price * quantity * (1 - discount))`, and active accounts require filtering out soft-deleted records). Fine-tuning encodes schema topology and business domain logic directly into adapter weights.
- **Serving Efficiency**: An 8B parameter model quantized to 4-bit runs with minimal VRAM (~5.5 GB), enabling deployment on cost-effective infrastructure or edge servers.

### Q2: Why QLoRA instead of Full Fine-Tuning or standard LoRA?
- **Hardware Accessibility & Zero Compute Budget**: Full fine-tuning of an 8B model in FP16 requires >64 GB VRAM across multiple enterprise GPUs (A100s). QLoRA allowed us to train the entire system for free using Kaggle’s 2x T4 GPUs (16 GB VRAM).
- **NormalFloat 4-bit (NF4)**: NF4 preserves information density for zero-mean, unit-variance normally distributed neural network weights better than standard 4-bit integer quantization.
- **Double Quantization (DQ)**: Quantizing quantization constants saves an additional 0.37 bits per parameter.
- **Full Model Quality Retention**: QLoRA matches 16-bit full fine-tuning performance while updating only 0.2% of trainable parameters ($r=16, \alpha=32$).

### Q3: Why Completion-Only Loss (`DataCollatorForCompletionOnlyLM`)?
- In standard autoregressive training, cross-entropy loss is computed across all tokens in the sequence (system instructions + schema DDL + natural language question + SQL output).
- The system instructions and schema context account for over **80% of total sequence tokens** (~1,400 tokens out of 1,700). Computing gradients over the schema forces the model to memorize the prompt itself, diluting the gradient signal.
- Masking everything up to the `[assistant]` response tag with `-100` forces the optimizer to update weights **strictly based on the SQL prediction tokens**.

### Q4: Why did Schema Retrieval (RAG) hurt performance in your ablation study?
- We built a lexical keyword and foreign-key join retriever (depth=2) designed to prune the 12 tables down to only the relevant subset per query.
- **The Empirical Result**: Accuracy dropped from **50.99% to 41.72%** on the fine-tuned model, and from 10.82% to 9.27% on the base model.
- **Root-Cause Analysis**: Our enterprise schema consists of 12 tables and 5,455 characters (~1,400 tokens). Modern LLMs comfortably handle context windows of 8k to 32k tokens without attention degradation. Dynamic retrieval introduced an imperfect retrieval recall rate (false negatives). When a critical join table was omitted by the retriever, the LLM had zero probability of generating a correct multi-table join.
- **Interview Takeaway**: *"We proved empirically that on schemas fitting within the model's native context window, schema pruning introduces selection noise and harms performance."*

### Q5: What is the Self-Correction Repair Loop, and what do its metrics reveal?
- When generated SQL fails either static AST validation or live PostgreSQL execution, the raw error traceback (e.g., `psycopg2.errors.UndefinedColumn: column "discount_pct" does not exist`) is wrapped into an iterative repair prompt.
- **The V2 Results**:
  - 25 queries failed initial generation and entered the repair loop.
  - **17 queries were repaired to execute without error (68.0% execution repair rate).**
  - **11 queries returned the exact ground-truth gold rows (44.0% true correctness rate).**
- **Crucial Nuance**: A query can be repaired into valid syntax that produces empty rows or wrong data. Tracking both *execution success* and *gold-row correctness* prevents reporting inflated metrics.

---

## 4. Key Metrics to Memorize

| Milestone / Configuration | Strict Execution Accuracy | Delta vs Baseline |
|:---|:---:|:---:|
| **Base Qwen3-8B (Prompt v1)** | 10.82% | Baseline floor |
| **Base Qwen3-8B + Schema Retrieval (v1)** | 9.27% | -1.55 pp (Retrieval penalty) |
| **Fine-Tuned Adapter (Prompt v1)** | 50.99% | +40.17 pp |
| **Fine-Tuned + Repair (v1)** | 52.10% | +41.28 pp |
| **Base Qwen3-8B (Prompt v2)** | 43.71% | +32.89 pp (Prompt engineering only) |
| **Fine-Tuned Adapter (Prompt v2)** | 68.43% | +24.72 pp over Base v2 |
| **Fine-Tuned + Repair (Prompt v2)** | **70.86%** | **Headline Benchmark (+27.15 pp over Base v2)** |

### Complexity Breakdown (Fine-Tuned + Repair v2)
- **Easy Queries** (single table, simple aggregates): **100.0%**
- **Medium Queries** (2-3 joins, GROUP BY, subqueries): **75.4%**
- **Hard Queries** (multi-table joins, complex date intervals, window CTEs): **62.5%**

---

## 5. Security & Production Engineering

### Multi-Tier Defense-in-Depth SQL Safety
1. **Abstract Syntax Tree (AST) Parsing**: Every candidate query is parsed via `sqlglot`. Malformed syntax or SQL injection attempts fail immediately before touching the network.
2. **Read-Only Enforced Classification**: AST nodes are inspected; only `SELECT` operations are whitelisted. Any statement containing `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, or `EXEC` is rejected with an HTTP 400 validation error.
3. **Identifier Whitelisting**: AST table and column expressions are verified against our 12-table schema dictionary. Queries referencing unauthorized tables or system catalogs (`pg_shadow`, `information_schema`) are blocked.
4. **Session-Level Read-Only Guarantee**: Every live PostgreSQL connection executes:
   ```sql
   SET default_transaction_read_only = on;
   SET statement_timeout = '30s';
   ```
   Even if malicious SQL bypassed application AST inspection, the database engine itself rejects write transactions and kills runaway queries after 30 seconds.

### Production Observability
- **LangSmith RunTree Spans**: Built custom non-blocking span tracing reporting prompt construction, LLM inference latency, AST validation time, database execution time, and repair loops.
- **Regional Endpoint Routing**: Solved a 403 API failure by handling regional routing (`https://apac.api.smith.langchain.com` vs default US endpoints) based on user API key prefixes.
- **`/metrics` Prometheus Endpoint**: Emits rolling operational statistics: overall execution pass rate, repair frequency, average latency per stage, and table access distribution.

---

## 6. How to Frame Challenging Interview Questions

#### *"How did you prevent benchmark data leakage?"*
> "We implemented group-level splitting by template family rather than random example splitting. We engineered 74 unique question/SQL templates across 12 business domains. Templates used in the 453-query test set were strictly excluded from the training and validation splits. Furthermore, the test set file has a cryptographic SHA-256 fingerprint (`ec7ddcae4f9d90d4`) verified inside our automated CI test suite on every commit. If a single byte changes, CI fails."

#### *"Why did you disable reasoning tokens (/no_think) on Qwen3-8B?"*
> "Qwen3 has an internal chain-of-thought mechanism that emits natural language reasoning tokens before producing the final code. During evaluation and production serving, interleaved reasoning strings cause parser failures and inflate inference token latency by 3-5x. By appending `/no_think` and configuring the generation prompt, we suppressed conversational preamble, achieving deterministic, single-block SQL generation."

#### *"What would you do next if given a dedicated budget and GPU cluster?"*
> 1. **Direct Preference Optimization (DPO)**: Pair correct SQL queries with plausible hallucinated counterparts to optimize the model away from subtle schema traps.
> 2. **Embedding-Based Hybrid Retrieval**: Replace naive keyword retrieval with dense bi-encoder embeddings combined with ColBERT re-ranking for schemas exceeding 100 tables.
> 3. **Speculative Decoding**: Use a lightweight 0.5B draft model to accelerate SQL token generation by 2.5x.
