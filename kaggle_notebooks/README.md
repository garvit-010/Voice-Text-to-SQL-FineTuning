# Kaggle Notebooks — Enterprise Text-to-SQL (QLoRA)

**Author:** Garvit Audichya  
**Model:** [`garvit-010/qwen3-8b-text2sql-qlora-v3`](https://huggingface.co/garvit-010/qwen3-8b-text2sql-qlora-v3)  
**Space (ZeroGPU Demo):** [`garvit-010/enterprise-text-to-sql`](https://huggingface.co/spaces/garvit-010/enterprise-text-to-sql)  
**GitHub:** [`garvit-010/Voice-Text-to-SQL-FineTuning`](https://github.com/garvit-010/Voice-Text-to-SQL-FineTuning)

This directory contains the Kaggle notebooks used to train and evaluate the QLoRA adapter on Kaggle's free GPU T4 tier.

---

## Notebooks

| # | Notebook | Purpose | Hardware | Output |
|---|---|---|---|---|
| **01** | [`01_qwen3_8b_qlora_finetune.ipynb`](./01_qwen3_8b_qlora_finetune.ipynb) | End-to-end QLoRA fine-tuning of Qwen3-8B | Kaggle GPU T4 x2 (~7 h) | `final_adapter/` + `predictions_v3.jsonl` |
| **02** | [`02_eval_predictions_generation.ipynb`](./02_eval_predictions_generation.ipynb) | Greedy generation over 453 test questions | Kaggle GPU T4 (~35 m) | `predictions_v3.jsonl` |
| **03** | [`03_self_correction_repair_loop.ipynb`](./03_self_correction_repair_loop.ipynb) | Single-turn repair loop on failing queries | Kaggle GPU T4 (~10 m) | `repairs.jsonl` |

---

## How to Run on Kaggle (Unattended)

1. **Upload Dataset:**
   - In Kaggle, go to **Datasets -> New Dataset**.
   - Upload the files from `dataset/sft/` (`train.jsonl`, `validation.jsonl`, `train_qlora.py`).
   - Title it `text2sql-sft-v3` and save.
2. **Import Notebook:**
   - In Kaggle, go to **Code -> New Notebook -> File -> Import Notebook**.
   - Select `01_qwen3_8b_qlora_finetune.ipynb`.
3. **Configure Settings:**
   - **Accelerator:** GPU T4 x2.
   - **Internet:** On (required to download base model weights `Qwen/Qwen3-8B`).
   - **Input:** Add your `text2sql-sft-v3` dataset.
4. **Launch Detached:**
   - Click **Save Version -> Save & Run All (Commit)**.
   - Close the browser tab. Kaggle runs the training to completion in the background.
5. **Download Artifacts:**
   - When finished, download `v3_results.zip` from the notebook's **Output** tab.
