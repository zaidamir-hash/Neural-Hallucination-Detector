"""Run the Phase 1 text-level audit and save artifacts."""
from pathlib import Path
from hallucination_detector.data.audit import analyze_records, save_json, plot_distribution
import json

RAW = Path("data/raw/qa_data.json")


def main():
    raw_text = RAW.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except Exception:
        lines = [l for l in raw_text.splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
    audit = analyze_records(data)
    save_json(audit, Path("results/metrics/halueval_text_audit.json"))
    # plot token-count distributions
    plot_distribution(audit["grounded"]["token_counts"], Path("results/figures/answer_length_grounded.png"), "Grounded Answer Token Count", "Tokens")
    plot_distribution(audit["hallucinated"]["token_counts"], Path("results/figures/answer_length_hallucinated.png"), "Hallucinated Answer Token Count", "Tokens")
    # lexical coverage distribution
    plot_distribution([int(x*100) for x in audit["grounded"]["lexical_coverages"]], Path("results/figures/lexical_coverage_grounded.png"), "Grounded Lexical Coverage (%)", "% Coverage")
    plot_distribution([int(x*100) for x in audit["hallucinated"]["lexical_coverages"]], Path("results/figures/lexical_coverage_hallucinated.png"), "Hallucinated Lexical Coverage (%)", "% Coverage")
    print("Audit complete. Outputs written to results/")


if __name__ == "__main__":
    main()
