# Neural Hallucination Detector

Project overview
----------------
A focused, reproducible experiment to detect hallucinated answers in question-answering examples using frozen sentence embeddings and lightweight classifiers.

Research question
-----------------
On paired HaluEval-QA examples, can a lightweight MLP trained on frozen sentence-embedding interaction features detect hallucinated answers more effectively than a cosine-similarity threshold and Logistic Regression?

Dataset
-------
Source: HaluEval QA subset (official repository: https://github.com/RUCAIBox/HaluEval). The QA subset uses the file `data/qa_data.json` from the official repo.

Data acquisition (Phase 1)
-------------------------
A reproducible acquisition script is provided:

- python scripts/acquire_halueval.py

The script downloads the revision-pinned raw file from the official repository and writes it to `data/raw/qa_data.json` (newline-delimited JSON / NDJSON). The download process computes a SHA-256 checksum and writes a tracked provenance record at `data/metadata/halueval_qa_provenance.json`.

Raw-data Git policy
-------------------
- Downloaded raw data is stored under `data/raw/` and is excluded from Git by `.gitignore`.
- Small provenance and manifest files live under `data/metadata/` and are tracked in Git (see `data/metadata/halueval_qa_provenance.json`).

Leakage-safe methodology
------------------------
Split original records before paired expansion; keep both answers from each source record in the same split. Use a single documented split seed.

Features
--------
Primary feature vector: four 384-d interaction blocks and seven scalar features; total dimension = 1,543.

Models
------
- Cosine-similarity threshold baseline
- Logistic Regression baseline on the primary features
- Lightweight PyTorch MLP with architecture 1543 -> 256 -> 64 -> 1

Evaluation protocol
-------------------
Primary metric: F1 for the hallucinated class (label = 1). Test set is locked during development; use validation data for model selection and threshold tuning.

Results
-------
TBD

Qualitative error analysis
--------------------------
TBD

Reproduction instructions
------------------------
See configs/default.yaml for the canonical configuration, and tests/ for Phase 0 and Phase 1 validation checks.

Phase 1 commands (data acquisition & audit)
------------------------------------------
- Acquire the official QA data and write provenance:
  - python scripts/acquire_halueval.py
- Validate original records and assign deterministic source IDs:
  - python scripts/validate_halueval.py
- Run the text-level audit and produce figures:
  - python scripts/audit_halueval.py
- Run duplicate audit:
  - python scripts/duplicate_audit.py

Provenance location
-------------------
Tracked provenance metadata is written to `data/metadata/halueval_qa_provenance.json`.

Phase 1 results (actual acquisition performed)
---------------------------------------------
- Official source: https://github.com/RUCAIBox/HaluEval (file: data/qa_data.json)
- Source revision: 207f479c555a68b8d2431631a95083b84b2accbe
- Local raw path: data/raw/qa_data.json
- SHA-256 checksum: 89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88
- Raw record count: 10,000
- Validation: 10,000 valid records, 0 excluded

Observed shortcut-risk signals (text-level, Phase 1)
---------------------------------------------------
Preliminary text-level audit (no embeddings) shows noticeable differences between grounded and hallucinated answers:
- Grounded answers are typically short (token mean ≈ 2.22, median 2), with high lexical overlap with the context (mean lexical coverage ≈ 0.897).
- Hallucinated answers are longer on average (token mean ≈ 10.99, median 9) and have lower lexical overlap with the context (mean lexical coverage ≈ 0.628).

These observations suggest potential shortcut risks based on answer length and lexical overlap. Embedding-based similarity checks are deferred to Phase 3.

Repository structure
--------------------
(unchanged) See PROJECT_PLAN.md for the recommended structure. Key folders include configs/, data/, src/, results/, and tests/.

Repository structure
--------------------
See PROJECT_PLAN.md for the recommended structure. Key folders:
- configs/: canonical experiment configuration
- data/: raw and processed data (excluded from git)
- src/: reusable code under src/hallucination_detector/
- tests/: phase-specific unit tests
- results/: generated figures, metrics, and error analyses

Limitations
-----------
TBD

References
----------
- HaluEval: https://github.com/RUCAIBox/HaluEval
- Project plan and methodology: PROJECT_PLAN.md
