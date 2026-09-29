"""Scoring metrics: accuracy, retrieval completeness, token efficiency.

- accuracy: normalized match between predicted answer and any gold answer.
  Gold answers are short strings (names/numbers), so we normalize case,
  punctuation, accents, and articles, and accept substring containment in
  either direction (the LLM may add a country code, e.g. "Chen Ding (CHN)").
- completeness: fraction of gold_doc_ids that appear in retrieved_doc_ids.
- tokens: total prompt+completion tokens for the pipeline run.
"""

from __future__ import annotations

import re
import unicodedata


def _normalize(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))  # strip accents
    s = s.lower().strip()
    s = re.sub(r"[\u2013\u2014\-]", " ", s)                    # dashes -> space
    s = re.sub(r"[^\w\s]", " ", s)                             # drop punctuation
    s = re.sub(r"\b(the|a|an|of|in|at|s)\b", " ", s)           # articles/'s
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _num(s: str) -> str | None:
    """Return the numeric value only if the string is essentially just a number
    (e.g. "5", "8", "3.14"). Titles that merely contain a year like
    "... 2008 ... marathon" are NOT numeric answers."""
    stripped = (s or "").strip().replace(",", "")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", stripped)
    return m.group() if m else None


def _num_in(s: str) -> str | None:
    """Extract the first number anywhere in a string (used on predictions,
    which may be phrased like 'The answer is 8.')."""
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return m.group() if m else None


def answer_matches(predicted: str, gold: list[str]) -> bool:
    pred_n = _normalize(predicted)
    if not pred_n:
        return False
    pred_num = _num_in(predicted)
    for g in gold:
        gold_n = _normalize(g)
        gold_num = _num(g)  # only set when gold is purely a number
        # numeric answers: compare the numbers directly
        if gold_num is not None:
            if pred_num is not None and float(pred_num) == float(gold_num):
                return True
            continue
        if not gold_n:
            continue
        if gold_n == pred_n:
            return True
        # containment either direction (handles added country codes, etc.)
        if gold_n in pred_n or pred_n in gold_n:
            return True
    return False


def completeness(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 1.0
    retrieved = set(retrieved_doc_ids)
    hit = sum(1 for d in gold_doc_ids if d in retrieved)
    return hit / len(gold_doc_ids)
