from __future__ import annotations

import re
import statistics
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def token_bag_f1(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    ref_set = set(ref)
    hyp_set = set(hyp)
    overlap = len(ref_set & hyp_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compression_ratio(input_text: str, response: str) -> float:
    in_len = max(len(input_text), 1)
    return len(response) / in_len


def token_recall(input_text: str, response: str) -> float:
    ref = set(tokenize(input_text))
    hyp = set(tokenize(response))
    if not ref:
        return 0.0
    return len(ref & hyp) / len(ref)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Word-level ROUGE-L F1 via LCS (lightweight, no extra deps)."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def summarize_row_metrics(input_text: str, response: str) -> dict[str, float]:
    return {
        "compression_ratio": round(compression_ratio(input_text, response), 4),
        "token_recall": round(token_recall(input_text, response), 4),
        "token_f1": round(token_bag_f1(input_text, response), 4),
        "rouge_l_f1": round(rouge_l_f1(input_text, response), 4),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows]
        out[key] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    return out
