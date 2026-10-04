# Training & Evaluation Walkthrough

Everything you need to walk someone through how this project was built, trained, and evaluated — in order.

---

## The One-Line Summary

> Fine-tune a free Kaggle GPU to turn Qwen3-8B into an enterprise SQL generator.
> Measure it honestly by *executing* every query against a real database.
> Strict execution accuracy: **43.71% to 68.43% to 70.86%** (base to fine-tuned to + self-correction).

---

## Phases at a Glance

| Phase | What Happens | Where it Lives |
|---|---|---|
| **1 — Database** | Build the real 12-table enterprise Postgres schema | `scripts/init_db.py`, `scripts/load_schema.py` |
| **2 — Synthetic Data** | Generate 3,000+ question/SQL pairs from templates | `scripts/generate_data.py` |
| **3 — Validation** | Execute every pair, reject bad ones, split train/val/test | `scripts/validate_dataset.py` |
| **4 — Baseline** | Run base Qwen3-8B, no fine-tuning. Freeze the floor. | `scripts/run_baseline.py` |
| **5 — SFT Dataset** | Convert validated pairs into chat-format for fine-tuning | `scripts/prepare_sft_dataset.py` |
| **6 — Upload to Kaggle** | Push train/val/script as a Kaggle Dataset | `training/kaggle/kaggle_upload_v3/` |
| **7 — QLoRA Training** | Train on Kaggle free GPU T4 (~7 hours, unattended) | `kaggle_notebooks/01_qwen3_8b_qlora_finetune.ipynb` |
| **8 — Export Eval Pack** | Pack the 453 test questions (no gold SQL) for GPU | `scripts/export_eval_pack.py` |
| **9 — Generate Predictions** | Run fine-tuned model over all 453 test questions | `kaggle_notebooks/02_eval_predictions_generation.ipynb` |
| **10 — Score Fine-Tuned** | Execute predictions, compare to gold on real DB | `scripts/score_finetuned.py` |
| **11 — Repair Loop** | Export failures, regenerate on GPU, rescore | `scripts/export_repair_pack.py` then `scripts/score_repair.py` |
| **12 — Ablation Report** | Assemble all configurations side-by-side | `scripts/ablation_report.py` |
| **13 — API Service** | FastAPI service exposing `/query` and `/voice` | `src/api/main.py` |
| **14 — Deploy** | Render (API) + HF Space (fine-tuned model demo) | `render.yaml`, `deploy/space/app.py` |

---

## Phase 1 — The Database

**Goal:** A real 12-table enterprise PostgreSQL schema. Not a toy.

```
scripts/init_db.py        # Creates roles, grants, schema (superuser)
scripts/load_schema.py    # Loads DDL, seeds reference data
scripts/generate_data.py  # Generates synthetic business data (~61 MB)
```

The schema mirrors a real enterprise: `customers`, `orders`, `order_items`, `employees`, `products`, `inventory`, `departments`, `salaries`, `payments`, `invoices`, `logistics`, `product_categories`.

**Why real Postgres?** Accuracy is measured by *executing* the query and comparing result rows. String matching is not used anywhere in this project.

---

## Phase 2 — Synthetic Dataset Generation

**Goal:** A benchmark where the test set cannot be contaminated by the internet.

```
scripts/generate_benchmark.py       # Generates question+SQL pairs from templates
dataset/generation/templates.py     # v1 templates
dataset/generation/templates_v2.py  # v2 templates (adds business glossary, harder questions)
```

Questions come from parameterized templates:

```
Template: "Which {dept} employees earn more than {salary}?"
SQL:      SELECT name FROM employees WHERE department = '{dept}' AND salary > {salary}
```

Every slot is filled with real values from the database. Questions are tagged by difficulty: `easy`, `medium`, `hard`, `very_hard`, `enterprise`. The test set is held out before any model ever runs.

---

## Phase 3 — Validation and Leakage-Free Splits

**Goal:** Make sure every example actually works and the splits are clean.

```
scripts/validate_dataset.py              # Runs the validation gauntlet
dataset/validated/VALIDATION_REPORT.md   # What passed, what was rejected and why
```

Each example goes through:
1. **Static analysis** — parseable SQL? One statement? Read-only? Real tables?
2. **Slot representation** — does the question mention the value the SQL filters on?
3. **Postgres execution** — does it actually run and return non-empty results?
4. **Duplicate check** — same question with conflicting SQL?

**Critical split decision:** Examples are split by *template equivalence group*, not by row. If three phrasings of the same query were split randomly, the model could score well by recognizing the pattern — not by understanding SQL. Group-level splits prevent this.

**Final dataset (v2):**

| Split | Examples | Templates |
|---|---|---|
| Train | 2,133 | 74 |
| Validation | 459 | 52 |
| **Test (held-out)** | **453** | **separate** |

---

## Phase 4 — The Frozen Baseline

**Goal:** Establish an honest floor before fine-tuning touches anything.

```
scripts/run_baseline.py                     # Runs base Qwen3-8B over all 453 test questions
experiments/v2/baseline/BASELINE_REPORT.md  # Frozen results
```

**Result (v2 benchmark):**

| Metric | Value |
|---|---|
| Test examples | 453 |
| **Strict execution accuracy** | **43.71%** |
| Executable SQL | 95.58% |
| Schema hallucination | 0.22% |

This number is **frozen**. It is the denominator for everything that follows.

**The prompt is fingerprinted.** `src/model/prompt.py` has a `prompt_fingerprint()` function. Every later step asserts the fingerprint matches. If the prompt drifts, the script aborts — not warns, aborts.

---

## Phase 5 — SFT Dataset Preparation

**Goal:** Convert validated pairs into a training format.

```
scripts/prepare_sft_dataset.py   # Builds train.jsonl and validation.jsonl
dataset/sft/manifest.json        # Records every decision made here
dataset/sft/SAMPLE.md            # Example of what a training record looks like
```

Each record becomes a chat message sequence:
```
[system]    {enterprise context + SQL conventions}
[user]      {schema + question + /no_think}
[assistant] {gold SQL only}
```

**Critical details:**
- Labels for the **prompt span are set to -100**. Only SQL tokens are trained on. Without this, ~97% of gradient signal goes into memorizing the schema (which is given at inference time anyway).
- The **test set fingerprint is checked** before writing. Leaked test examples would make the score measure memorization.
- The **prompt fingerprint is asserted** — byte-identical to the baseline.
- `/no_think` is appended. Qwen3-8B has a thinking mode. The baseline was measured with thinking off. Training with it on would measure something else.

---

## Phase 6 — Upload to Kaggle

**Goal:** Get training files to the GPU without committing secrets.

```
training/kaggle/kaggle_upload_v3/
    train.jsonl           # 2,133 training examples
    validation.jsonl      # 459 validation examples
    train_qlora.py        # The actual training script
    eval_pack.jsonl       # 453 test questions (for evaluation after training)
    prompt_module.py      # Exact prompt the baseline used
    schema_context.txt    # Rendered schema, same as baseline
```

The training script is part of the dataset so the notebook is self-contained — no external dependencies that could vanish mid-run.

---

## Phase 7 — QLoRA Fine-Tuning on Kaggle

**Goal:** Train the adapter. Close the browser. Come back to a finished model.

```
kaggle_notebooks/01_qwen3_8b_qlora_finetune.ipynb     # The training notebook
training/kaggle/kaggle_upload_v3/train_qlora.py        # The training script the notebook calls
models/finetuned_v2/MODEL_CARD.md                      # Final model card
models/finetuned_v2/training_config.json               # Every hyperparameter
models/finetuned_v2/training_metrics.json              # Loss curve + final eval
```

**Why Kaggle over Colab?** "Save & Run All" runs the notebook completely detached — you close the browser and Kaggle emails you when it is done. Colab requires a live browser tab and reclaims sessions unpredictably. A 7-hour training run cannot survive that.

**Hardware:** Kaggle free tier, single Nvidia T4 (16 GB VRAM, 30 GPU h/week).

**Fine-tuning configuration:**

| Parameter | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-8B` rev `b968826d9c46` | Pinned revision — same weights the baseline used |
| Quantization | 4-bit NF4 (QLoRA) | At bf16 = 16.4 GB, does not fit T4. At 4-bit = 4.4 GB. |
| LoRA rank | r=16, alpha=32 | 43.6M trainable params (0.917% of model) |
| LoRA targets | All attention + MLP projections | More coverage than attention-only |
| Loss | Completion-only (SQL tokens only) | Prompt is 97% schema — training on it wastes gradient |
| Epochs | 1 | Train loss 0.083, eval loss 0.145 — converged cleanly |
| Batch | 1 + gradient accumulation 16 | Effective batch 16; batch 2 ran out of memory |
| Precision | fp16 | T4 is Turing — no bf16 hardware support |
| Training time | **7 h 05 m** | 266 optimizer steps at ~75 s/step |

---

## Phase 8 — Export Eval Pack

**Goal:** Send test questions to the GPU without leaking gold answers.

```
scripts/export_eval_pack.py          # Creates evalpack.zip
experiments/v2/finetuned/evalpack/   # What was sent to Kaggle
```

Both the prompt fingerprint and schema fingerprint are asserted before the export writes anything. If either has drifted from the frozen baseline values, the script aborts. The model cannot copy an answer it was never given — this is a structural guarantee.

---

## Phase 9 — Generate Predictions on Kaggle

**Goal:** Run the fine-tuned model over all 453 test questions.

```
kaggle_notebooks/02_eval_predictions_generation.ipynb
```

The notebook:
1. Loads the eval pack from Kaggle input.
2. Loads `Qwen/Qwen3-8B` in 4-bit + attaches `garvit-010/qwen3-8b-text2sql-qlora-v3`.
3. Generates SQL with **greedy decoding** (temperature=0, deterministic).
4. Writes **raw model output** to `predictions_v3.jsonl` — not extracted SQL.
5. Packages as a downloadable zip from the Output tab.

**Why raw output, not extracted SQL?** `extract_sql()` runs on the laptop, using the same function as the baseline. A second copy on the GPU host could drift and silently change the score.

---

## Phase 10 — Score the Fine-Tuned Model

**Goal:** Execute predictions against the real database.

```
scripts/score_finetuned.py                    # Runs predictions through scoring harness
experiments/v2/finetuned/FINETUNED_REPORT.md  # Results
experiments/v2/finetuned/summary.json         # Machine-readable (for ablation_report.py)
```

```powershell
env\Scripts\python.exe scripts/score_finetuned.py --version v2 --predictions "path/to/predictions_v3.jsonl"
```

The harness: extract SQL → static validate (sqlglot) → execute (read-only Postgres) → compare result fingerprints.

**Result (v2, fine-tuned, no repair):**

| Metric | Baseline | Fine-Tuned | Change |
|---|---:|---:|---:|
| Strict execution accuracy | 43.71% | **68.43%** | **+24.72 pp** |
| Executable SQL | 95.58% | 94.48% | -1.10 pp |
| Schema hallucination | 0.22% | 3.75% | +3.53 pp |

The hallucination uptick is expected: the model learned column naming conventions so precisely it occasionally invented a column name with the right structure. The accuracy gain is real.

---

## Phase 11 — Self-Correction Repair Loop

**Goal:** Can the model fix its own failures when shown the error?

```
scripts/export_repair_pack.py                           # Packs failing queries + live error text
kaggle_notebooks/03_self_correction_repair_loop.ipynb   # Regenerates SQL on GPU
scripts/score_repair.py                                 # Folds repairs in, rescores all 453
experiments/v2/repair/REPAIR_REPORT.md                  # Results
```

The repair prompt shows the model:
- The original question
- The SQL it wrote
- The exact error the database returned
- **No gold SQL** — the model must reason, not copy

**Result:**

| Metric | Fine-Tuned | + Repair | Change |
|---|---:|---:|---:|
| Strict execution accuracy | 68.43% | **70.86%** | +2.43 pp |
| Executable SQL | 94.48% | **98.23%** | +3.75 pp |
| Schema hallucination | 3.75% | **0.66%** | -3.09 pp |

**Repair statistics (25 failures retried):**
- 17 now execute (68% repair success)
- 11 return correct rows (44% repair correctness — the honest number)
- 5 returned identical SQL (model could not improve it)
- Mean repair latency: 15,448 ms

Two rates are reported on purpose. A repair that turns a crash into a confident wrong answer counts as *success* but not *correctness*. Reporting only success would overstate what repair achieves.

---

## Phase 12 — Ablation Report

**Goal:** All configurations side by side on the same test set.

```
scripts/ablation_report.py   # Collects all summary.json files, renders the table
experiments/v2/ABLATION.md   # The final table
```

**Final ablation (v2 benchmark, 453 questions):**

| Metric | 1. Base | 3. Fine-Tuned | 5. FT + Repair |
|---|---:|---:|---:|
| Strict execution accuracy | 43.71% | 68.43% | **70.86%** |
| Executable SQL | 95.58% | 94.48% | **98.23%** |
| Schema hallucination | **0.22%** | 3.75% | 0.66% |

**By difficulty:**

| Tier | n | Base | Fine-Tuned | + Repair |
|---|---:|---:|---:|---:|
| Easy | 126 | 92.9% | 100.0% | **100.0%** |
| Medium | 63 | 69.8% | 96.8% | **96.8%** |
| Hard | 144 | 7.6% | 54.9% | **62.5%** |
| Very Hard | 36 | 0.0% | 13.9% | **13.9%** |
| Enterprise | 84 | 31.0% | 46.4% | **46.4%** |

Configurations 2 and 4 (schema retrieval) are listed as **not measured**. The v1 experiment showed retrieval *hurt* accuracy on this 12-table schema (10.82% to 9.27%) — a result worth documenting, not hiding.

---

## Phase 13 — Production API

**Goal:** A real HTTP service anyone can hit.

```
src/api/main.py           # FastAPI app: /query, /voice, /health, /schema, /metrics
src/api/service.py        # The pipeline: generate -> validate -> execute -> repair
src/api/backends.py       # Backends: hf_space (fine-tuned), hf (base), local, stub
src/api/schemas.py        # Request/response types
src/api/observability.py  # LangSmith tracing + Prometheus metrics
```

The `/query` flow: question → prompt v2 → model (HF Space ZeroGPU) → SQL → static validation → Postgres execution → [fail] repair loop → rows + metadata.

Safety layers (none depend on model behavior):
1. Database role is not a superuser.
2. Every session is read-only with a statement timeout.
3. SQL is statically rejected unless it is a single SELECT over real tables.

The API returns **HTTP 200 even when SQL fails**. `ok: false` in the body carries that signal. HTTP 5xx is reserved for the service itself being broken.

---

## Phase 14 — Deploy

**Goal:** Live, publicly accessible, free.

```
render.yaml              # Render blueprint: FastAPI on Render free tier
deploy/space/app.py      # Gradio UI on HF Spaces (ZeroGPU, fine-tuned model)
scripts/push_space.py    # Pushes deploy/space/ to garvit-010/enterprise-text-to-sql
scripts/upload_model.py  # Publishes adapter to garvit-010/qwen3-8b-text2sql-qlora-v3
```

**Why HF Space for the model?** Render's free tier has no GPU. The HF Space gets a free ZeroGPU A100 slice per request. Render calls it via HTTP — the fine-tuned model answers every `/query`, not just the Space UI.

**Total cost: $0.** Neon (free tier, no expiry), Render (750 h/month free), HF ZeroGPU (free daily quota), Kaggle GPU (30 h/week free).

---

## All Files: What Does What

### Notebooks (run on Kaggle GPU)

| File | Purpose |
|---|---|
| `kaggle_notebooks/01_qwen3_8b_qlora_finetune.ipynb` | End-to-end QLoRA training + eval |
| `kaggle_notebooks/02_eval_predictions_generation.ipynb` | Standalone: generate predictions for 453 questions |
| `kaggle_notebooks/03_self_correction_repair_loop.ipynb` | Standalone: regenerate failed queries |
| `training/kaggle/kaggle_upload_v3/train_qlora.py` | The training script (called by notebook 01) |

### Scripts (run locally)

| Script | Phase | Purpose |
|---|---|---|
| `scripts/init_db.py` | 1 | Create Postgres roles and schema |
| `scripts/load_schema.py` | 1 | Load DDL and seed reference data |
| `scripts/generate_data.py` | 2 | Generate synthetic business data |
| `scripts/generate_benchmark.py` | 2 | Generate question/SQL pairs from templates |
| `scripts/validate_dataset.py` | 3 | Validate + split into train/val/test |
| `scripts/run_baseline.py` | 4 | Run base model benchmark, freeze results |
| `scripts/prepare_sft_dataset.py` | 5 | Convert to chat format for fine-tuning |
| `scripts/export_eval_pack.py` | 8 | Pack test questions (no gold) for Kaggle |
| `scripts/score_finetuned.py` | 10 | Execute predictions, compare to gold |
| `scripts/export_repair_pack.py` | 11 | Pack failing queries + errors for Kaggle |
| `scripts/score_repair.py` | 11 | Fold repairs in, rescore all 453 |
| `scripts/ablation_report.py` | 12 | Assemble all configurations side-by-side |
| `scripts/failure_analysis.py` | 12 | Classify why remaining failures failed |
| `scripts/upload_model.py` | 14 | Publish adapter to HuggingFace Hub |
| `scripts/upload_dataset.py` | 14 | Publish benchmark to HuggingFace Hub |
| `scripts/push_space.py` | 14 | Push Gradio Space to HuggingFace |

### Source Code (the live pipeline)

| File | What it does |
|---|---|
| `src/model/prompt.py` | Prompt v1 (fingerprint `8288e41a496531a9`) |
| `src/model/prompt_v2.py` | Prompt v2 + business glossary (fingerprint `4e72cc5f722ce436`) |
| `src/model/repair_prompt.py` | Repair prompt: failed SQL + error → fixed SQL |
| `src/model/base_model.py` | HF Inference Provider HTTP client |
| `src/model/schema_context.py` | Renders live schema from Postgres into prompt string |
| `src/sql/validator.py` | Static SQL validation (sqlglot) |
| `src/sql/executor.py` | Read-only Postgres execution with timeout |
| `src/api/backends.py` | HFSpaceModel, HFInferenceModel, LocalAdapterModel, StubModel |
| `src/api/main.py` | FastAPI endpoints |
| `src/api/service.py` | End-to-end pipeline logic |
| `src/api/observability.py` | LangSmith tracing + Prometheus |

### Experiment Results

| File | What it records |
|---|---|
| `experiments/v2/baseline/BASELINE_REPORT.md` | Config 1: base model, **43.71%** |
| `experiments/v2/finetuned/FINETUNED_REPORT.md` | Config 3: fine-tuned, **68.43%** |
| `experiments/v2/repair/REPAIR_REPORT.md` | Config 5: fine-tuned + repair, **70.86%** |
| `experiments/v2/ABLATION.md` | All three side-by-side |
| `experiments/v2/FAILURE_ANALYSIS.md` | Why the remaining 29.14% failed |
| `models/finetuned_v2/training_config.json` | Every hyperparameter |
| `models/finetuned_v2/training_metrics.json` | Loss curve (train 0.083, eval 0.145) |
| `dataset/sft/manifest.json` | Every decision in Phase 5 |
| `dataset/validated/VALIDATION_REPORT.md` | What was rejected and why |

---

## The Numbers to Know

```
43.71%  ->  68.43%  ->  70.86%
```

- **43.71%** — Base Qwen3-8B, prompt v2, 453 held-out questions. The frozen floor.
- **68.43%** — Fine-tuned adapter (QLoRA, 1 epoch, T4, 7 h), same test set. +24.72 pp.
- **70.86%** — Fine-tuned + self-correction repair loop. +2.43 pp more.
- Measured by **executing every query** against real PostgreSQL. Not string matching. Not LLM judging.
