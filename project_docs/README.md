# Project Documentation & Interview Portfolio

Welcome to the centralized documentation directory for the **Enterprise Voice & Text-to-SQL Fine-Tuning** project. This folder contains all architectural walkthroughs, technical interview preparation material, conceptual glossaries, and consolidated benchmark reports.

---

## Directory Navigation

| File / Folder | Description | Primary Audience |
|:---|:---|:---|
| **[INTERVIEW_GUIDE.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/INTERVIEW_GUIDE.md)** | End-to-end interview playbook: 60-second pitch, system architecture, challenging interview Q&As, and key numbers to memorize. | Technical Interviewers, Hiring Managers |
| **[TRAINING_WALKTHROUGH.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/TRAINING_WALKTHROUGH.md)** | Step-by-step 14-phase engineering guide: database setup, synthetic generation, validation, Kaggle QLoRA fine-tuning, evaluation, repair loop, and deployment. | ML Engineers, Peer Reviewers |
| **[BUZZWORDS_AND_CONCEPTS.md](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/BUZZWORDS_AND_CONCEPTS.md)** | Deep conceptual dictionary defining buzzwords with rigorous technical precision: Strict Execution, Self-Correction, QLoRA, NF4, Double Quantization, Completion Loss, AST safety, etc. | Technical Deep Dives, Systems Design |
| **[reports/](file:///d:/Projects/Voice&Text-to-SQL-FineTuning/project_docs/reports/)** | Centralized directory housing all 12 experimental reports, ablation studies, and failure analyses for v1 and v2. | Benchmark Verification & Audits |

---

## Quick Reference: Project Headline Metrics

- **Base Model (Qwen3-8B) + Prompt v2**: `43.71%` Strict Execution Accuracy
- **Fine-Tuned Adapter (QLoRA) + Prompt v2**: `68.43%` Strict Execution Accuracy (+24.72 pp lift)
- **Fine-Tuned + Self-Correction Repair Loop**: `70.86%` Strict Execution Accuracy (**Headline Result**)
- **Easy Query Tier**: `100.0%`
- **Medium Query Tier**: `75.4%`
- **Hard Query Tier**: `62.5%`
- **Test Set**: 453 queries, sealed with SHA-256 fingerprint `ec7ddcae4f9d90d4`

---

## Note on Git Tracking & Private Documents

If you have personal notes or confidential documents you wish to keep private to your local environment:
1. You can create a file like `private_notes.md` or a folder like `private/` inside `project_docs/`.
2. Add `project_docs/private/` or `project_docs/private_*.md` to your `.gitignore` file before running `git push`.
3. All other public documents in this folder (`INTERVIEW_GUIDE.md`, `TRAINING_WALKTHROUGH.md`, `BUZZWORDS_AND_CONCEPTS.md`, and `reports/`) are ready and sanitized for pushing to GitHub / origin.
