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
See configs/default.yaml for the canonical configuration, and tests/ for Phase 0–5 validation checks.

Environment setup
-----------------
- Python 3.12.0 (dedicated .venv at project root)
- Install: `pip install -r requirements.txt`
- Key packages: torch 2.14.0+cpu, sentence-transformers 6.0.1, scikit-learn 1.9.0

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

Phase 2 commands (leakage-safe splitting)
-----------------------------------------
- Create the original-record split manifest:
  - python scripts/create_split_manifest.py
- Expand records only after split assignment:
  - python scripts/expand_split_records.py
- Validate leakage and pairing invariants:
  - python scripts/validate_split_leakage.py

Phase 3 commands (frozen embeddings & features)
----------------------------------------------
- Run the full Phase 3 pipeline (encoding, chunking, feature generation, scaling):
  - python scripts/run_phase3.py
- Explicitly replace incompatible or stale embedding caches:
  - python scripts/run_phase3.py --force-recompute
- Generate the descriptive semantic-similarity audit from training data only:
  - python scripts/audit_train_similarity.py

Phase 3 methodology
-------------------
- Normalize each question, candidate answer, and reference context with `" ".join(text.split())`: require a string, strip outer whitespace, and collapse consecutive Unicode whitespace to one ordinary space. Apply this before deduplication, encoding, chunking, and scalar coverage calculation. Raw/processed source files, source/example IDs, splits, and labels are preserved.
- Preprocessing cache version: `2.0.0`; caches from the previous preprocessing version are rejected and require explicit regeneration with `--force-recompute`.
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (frozen, 384-d, L2-normalized)
- Context chunking: 192 model tokens per chunk, 32-token overlap, tokenizer-aware
- Primary feature vector (dim=1,543):
  - 4 interaction blocks (384 each): |a - c_a|, a * c_a, |q - a|, q * a
  - 7 scalar features: max answer-context cosine, mean top-3 answer-context cosine, question-answer cosine, max question-context cosine, answer content-token coverage, answer numeric-token coverage, numeric-token presence
- Scaling: StandardScaler fit on training data only, applied to all splits

Phase 3 artifacts
-----------------
- data/features/{train,validation,test}/features.npy — raw features (N, 1543)
- data/features/{train,validation,test}/features_scaled.npy — scaled features (N, 1543)
- data/features/{train,validation,test}/labels.npy — binary labels
- data/features/{train,validation,test}/example_ids.json — stable example IDs
- data/features/{train,validation,test}/source_ids.json — source record IDs
- data/features/{train,validation,test}/metadata.json — per-split stats
- data/features/schema.json — feature schema with block boundaries and scalar indices
- data/scaler/scaler.npz — fitted scaler parameters (mean, scale)
- data/embedding_cache/ — compatible question, answer, context, and chunk embeddings with metadata

Phase 3 cache policy
--------------------
- Compatible embedding caches are validated and reused.
- Incomplete or incompatible caches fail clearly instead of being silently mixed or replaced.
- `--force-recompute` explicitly refreshes all embedding caches.
- Embedding caches, feature matrices, fitted scaler artifacts, and `.venv` are excluded from Git.

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

These observations suggest potential shortcut risks based on answer length and lexical overlap. Phase 3 now supplies the frozen-embedding similarity features used by the primary models.

Train-only semantic-similarity audit (Phase 3)
---------------------------------------------
`python scripts/audit_train_similarity.py` compares the unscaled maximum answer-context-chunk cosine by label using only `data/features/train/features.npy` and training labels. It writes aggregate counts, means, population standard deviations, quartiles, and ranges to `results/metrics/halueval_train_semantic_audit.json` and a comparison histogram to `results/figures/train_answer_context_cosine.png`.

After normalized regeneration, each training class contains 7,000 examples. Grounded answers have mean cosine 0.3611 (median 0.3731; population SD 0.2111), while hallucinated answers have mean 0.5628 (median 0.5838; population SD 0.1882). Hallucinated answers therefore have higher similarity on average in this training split, contrary to the cosine baseline's original expected direction. The distributions overlap substantially; the earlier answer-length differences are a possible contributing benchmark artifact, not an established explanation. The feature design remains unchanged; the user subsequently authorized validation-only selection between both cosine directions in Phase 4.

This descriptive audit does not calculate test performance or change the locked feature design. Differences may reflect grounding and benchmark shortcuts; they do not establish causality or generalization.

Phase 4 baselines and negative control (validation only)
-------------------------------------------------------
Run from the repository root: `.venv\Scripts\python.exe scripts/run_phase4.py`.
Run all tests: `.venv\Scripts\python.exe -m pytest tests -q`.

The Phase 4 workflow opens training and validation features only (14,000 and 3,000 paired examples, respectively). It refuses test split inputs. Training rows fit the models; validation alone selects cosine direction/threshold, Logistic Regression regularization, and probability thresholds. No model is refitted on combined train/validation data. These are selection-set results, not final test estimates.

- Cosine: unscaled `max_answer_context_cosine` at column 1536. Both directions are searched as explicitly authorized after the training audit. The selected rule is **Hallucinated when similarity >= 0.2736169099807739**, with continuous score equal to similarity for ROC-AUC. Selecting the higher-similarity direction may reflect HaluEval construction artifacts; it does not establish a general rule about grounding.
- Primary Logistic Regression: L2, shared saved scaled 1,543 features, `C` in `[0.01, 0.1, 1, 10]`, no class weights, deterministic `lbfgs`, seed 42, maximum 5,000 iterations, tolerance `1e-5`, one BLAS thread. Selected `C=0.01`, probability threshold `0.48363664746284485`.
- Negative control: separate L2 Logistic Regression with answer character count, whitespace-token count, lightweight sentence count, numeric-token presence/count, and Unicode punctuation count. Features use original processed answer strings. A separate StandardScaler fits on training control features only. Question/context data, semantic features, coverage, and IDs are excluded as predictors. Selected `C=0.01`, probability threshold `0.46458989417782254`. This diagnostic is not a primary model.

Thresholds use `score >= threshold`; lower-cosine direction uses `score=-similarity`, equivalent to `similarity <= threshold`. Candidates are distinct validation scores, a reference (0.5 for probabilities, 0 for signed cosine), and the next representable value above the maximum. Ties prefer proximity to the reference, then smaller threshold. Direction ties prefer the original lower-similarity direction. Logistic model selection orders by highest validation F1, highest validation ROC-AUC, smaller C, threshold closest to 0.5, then smaller threshold.

| Approach (validation only) | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Cosine | 0.6457 | 0.5948 | 0.9140 | 0.7206 | 0.7542 |
| Primary Logistic Regression | 0.9493 | 0.9469 | 0.9520 | 0.9495 | 0.9889 |
| Answer-only negative control | 0.9537 | 0.9516 | 0.9560 | 0.9538 | 0.9767 |

All eight Logistic Regression candidate fits converged. Primary iterations for C `[0.01, 0.1, 1, 10]`: `[35, 60, 109, 132]`; control iterations: `[16, 19, 22, 25]`. Continuous probabilities, not thresholded predictions, determine Logistic Regression ROC-AUC.

The strong answer-only result, together with the large answer-length difference (Phase 1 token means approximately 2.22 versus 10.99), is a serious benchmark shortcut concern. High validation performance cannot be attributed solely to context grounding. These results do not establish real-world hallucination detection or a final model ranking.

Tracked configuration, complete C searches, cosine direction search, validation metrics/confusion matrices, control feature schema/scaler parameters, library versions, and input provenance are in `results/metrics/phase4_validation.json`. Fitted estimators, both scaler parameter sets, and selected rules are saved without raw text in ignored `checkpoints/phase4/baselines.joblib`. Only trusted local joblib checkpoints should be loaded; the loader checks feature schema and preprocessing compatibility. No final test evaluation, MLP, shuffled-context evaluation, or error analysis has been run in Phase 4.

Phase 5 MLP training (validation only)
-------------------------------------
From the repository root, with the existing Phase 3 artifacts and Phase 4 validation report:

```powershell
.venv\Scripts\python.exe scripts/run_phase5.py
.venv\Scripts\python.exe -m pytest tests -q
```

The runner uses only the saved training-scaled 1,543-dimensional training and validation matrices (14,000 and 3,000 examples). Before training, it verifies the canonical schema, preprocessing version, dataset/source and split provenance, embedding metadata, IDs and labels, and unchanged Phase 4 input checksums. Phase 4 already verified the scaler against training moments; Phase 5 checks its identity and verifies both saved scaled matrices against its transformation. It never fits another scaler, regenerates embeddings, or opens test features/labels.

The locked network is `1543 -> Linear(256) -> ReLU -> Dropout(0.20) -> Linear(64) -> ReLU -> Dropout(0.10) -> Linear(1)`, returning raw logits. Training uses unweighted BCEWithLogitsLoss, AdamW, numeric learning rate `0.001`, weight decay `0.0001`, batch size 128, maximum 30 epochs, and patience 5. Both epoch losses are weighted by batch sample counts, including incomplete final batches. Validation uses evaluation/inference mode. Strictly lower validation loss saves the best checkpoint; equal losses retain the earlier epoch. Each run restores that checkpoint before sigmoid prediction and validation-only F1 threshold selection using the existing Phase 4 utility and tie rules.

The three runs seed Python, NumPy, PyTorch, and an independent training DataLoader generator. Only training batches shuffle. This execution used CPU, one PyTorch compute thread, no loader workers, deterministic algorithms enabled, cuDNN deterministic enabled and benchmark disabled. Exact library versions, execution settings, code hashes, and input hashes are recorded in the summary. Reproducibility applies to the same software/hardware settings; bitwise identity across platforms is not guaranteed.

**Validation selection-set results, not final test estimates:**

| Seed | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC | Threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.940000 | 0.937666 | 0.942667 | 0.940160 | 0.984600 | 0.512878954410553 | 2 |
| 202 | 0.940333 | 0.941216 | 0.939333 | 0.940274 | 0.984942 | 0.3784696161746979 | 2 |
| 303 | 0.943667 | 0.944556 | 0.942667 | 0.943610 | 0.984750 | 0.5508548617362976 | 2 |
| Mean | 0.941333 | 0.941146 | 0.941556 | 0.941348 | 0.984764 | — | — |
| Sample SD | 0.002028 | 0.003446 | 0.001925 | 0.001960 | 0.000171 | — | — |

Standard deviations use `ddof=1` across the three seeds. All three stopped after epoch 7, following five epochs without improvement after epoch 2. Best validation BCE losses were 0.1642051882, 0.1608461339, and 0.1676170768 for seeds 101, 202, and 303, respectively. Confusion matrices (rows=true, columns=predicted, labels `[0,1]`) were `[[1406,94],[86,1414]]`, `[[1412,88],[91,1409]]`, and `[[1417,83],[86,1414]]`.

The representative seed is **202**, selected by median validation F1; exact F1 ties choose the smaller seed. All three models are preserved. These MLP validation scores are below the Phase 4 primary Logistic Regression and answer-only negative control scores. The strong answer-only control and documented answer-length and semantic-similarity differences remain serious shortcut concerns: these results alone do not establish context grounding, real-world generalization, or a final model ranking. No architecture or hyperparameters were changed in response to results.

Artifacts:

- `results/metrics/phase5_validation.json`: all seed metrics, thresholds, best epochs, aggregate mean/sample SD, selection policies, configuration, library versions, and provenance.
- `results/metrics/phase5/seed_{101,202,303}_history.json`: complete epoch histories, tracked with the summary.
- `results/figures/phase5_losses.png`: tracked training/validation loss curves for all seeds.
- `checkpoints/phase5/mlp_seed_{101,202,303}.pt`: ignored independent best state dictionaries, thresholds, configuration, and compatibility metadata. Reload uses PyTorch's restricted `weights_only=True` loader and checks schema, preprocessing, scaler and input provenance.

![Phase 5 training and validation loss curves](results/figures/phase5_losses.png)

The existing encoder metadata records its revision as unavailable (`null`); Phase 5 preserves that limitation and does not reload the encoder. The run also emitted a nonfatal Matplotlib font-cache permission warning; the loss figure was generated successfully. Phase 5 stops at validation: final test evaluation, shuffled-context analysis, and qualitative error analysis remain future phases.

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
