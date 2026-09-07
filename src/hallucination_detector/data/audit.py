"""Text-level shortcut audit utilities for HaluEval QA.

Produces metrics and figures for grounded and hallucinated answers without using any
sentence embedding model. Embedding-based checks are deferred to Phase 3.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from collections import Counter
import matplotlib.pyplot as plt


def sentence_count(text: str) -> int:
    # Very lightweight rule: split on sentence-ending punctuation . ? ! and filter
    if not text:
        return 0
    parts = re.split(r"[\.\?!]+", text)
    return sum(1 for p in parts if p.strip())


def tokens_whitespace(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"\s+", text.strip()) if t]


def content_tokens(text: str) -> List[str]:
    # Simple tokenization: split, lowercase, strip punctuation
    toks = tokens_whitespace(text)
    cleaned = [re.sub(r"(^\W+|\W+$)", "", t).lower() for t in toks]
    return [t for t in cleaned if t]


def numeric_tokens(text: str) -> List[str]:
    toks = tokens_whitespace(text)
    return [t for t in toks if re.search(r"\d", t)]


def lexical_coverage(answer: str, context: str) -> float:
    a_tokens = set(content_tokens(answer))
    if not a_tokens:
        return 0.0
    c_tokens = set(content_tokens(context))
    covered = sum(1 for t in a_tokens if t in c_tokens)
    return covered / len(a_tokens)


def exact_normalized_occurrence(answer: str, context: str) -> bool:
    a = re.sub(r"\s+", " ", answer.strip().lower())
    c = re.sub(r"\s+", " ", context.strip().lower())
    return a in c


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # records are expected to be dicts with required fields
    grounded_stats = []
    hall_stats = []

    for rec in records:
        q = rec.get("question", "")
        kg = rec.get("knowledge", "")
        ra = rec.get("right_answer", "")
        ha = rec.get("hallucinated_answer", "")

        for label, ans in (("grounded", ra), ("hallucinated", ha)):
            entry = {
                "char_count": len(ans),
                "token_count": len(tokens_whitespace(ans)),
                "sentence_count": sentence_count(ans),
                "numeric_tokens": numeric_tokens(ans),
                "numeric_token_count": len(numeric_tokens(ans)),
                "has_numeric": 1 if re.search(r"\d", ans) else 0,
                "lexical_coverage": lexical_coverage(ans, kg),
                "exact_in_context": exact_normalized_occurrence(ans, kg),
            }
            if label == "grounded":
                grounded_stats.append(entry)
            else:
                hall_stats.append(entry)

    def summarize(list_of):
        return {
            "count": len(list_of),
            "char_counts": [e["char_count"] for e in list_of],
            "token_counts": [e["token_count"] for e in list_of],
            "sentence_counts": [e["sentence_count"] for e in list_of],
            "numeric_token_counts": [e["numeric_token_count"] for e in list_of],
            "has_numeric_counts": sum(e["has_numeric"] for e in list_of),
            "lexical_coverages": [e["lexical_coverage"] for e in list_of],
            "exact_in_context_counts": sum(1 for e in list_of if e["exact_in_context"]),
        }

    gsum = summarize(grounded_stats)
    hsum = summarize(hall_stats)
    return {"grounded": gsum, "hallucinated": hsum}


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def plot_distribution(values: List[int], out_path: Path, title: str, xlabel: str):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, color="#2c7fb8", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=str, default="data/raw/qa_data.json")
    p.add_argument("--out-prefix", type=str, default="results/metrics/halueval_text_audit")
    args = p.parse_args()

    raw = Path(args.raw)
    data = json.loads(raw.read_text(encoding="utf-8"))
    audit = analyze_records(data)
    save_json(audit, Path(f"{args.out_prefix}.json"))
    # example figures
    plot_distribution(audit["grounded"]["token_counts"], Path("results/figures/answer_length_grounded.png"), "Grounded Answer Token Count", "Tokens")
    plot_distribution(audit["hallucinated"]["token_counts"], Path("results/figures/answer_length_hallucinated.png"), "Hallucinated Answer Token Count", "Tokens")
