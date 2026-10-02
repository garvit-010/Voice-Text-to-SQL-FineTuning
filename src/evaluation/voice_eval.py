"""Voice Speech-to-Text Word Error Rate (WER) Evaluation Harness.

Uses jiwer to calculate WER, MER (Match Error Rate), and WIL (Word Information Lost)
over reference vs transcribed voice queries.
"""

from __future__ import annotations

from typing import Any, Dict, List
import jiwer


def compute_wer_metrics(reference_texts: List[str], hypothesis_texts: List[str]) -> Dict[str, float]:
    """Compute Word Error Rate metrics across lists of reference & transcribed texts."""
    if not reference_texts or not hypothesis_texts or len(reference_texts) != len(hypothesis_texts):
        raise ValueError("reference_texts and hypothesis_texts must be non-empty and equal length")

    out = jiwer.process_words(reference_texts, hypothesis_texts)

    return {
        "word_error_rate_wer": round(float(out.wer) * 100, 2),
        "match_error_rate_mer": round(float(out.mer) * 100, 2),
        "word_info_lost_wil": round(float(out.wil) * 100, 2),
        "sample_count": len(reference_texts),
    }
