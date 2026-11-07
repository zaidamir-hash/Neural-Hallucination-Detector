# Phase 8 verification record

Initial verification: **2026-11-05**. Independent run: **2026-11-07**. Starting repository: clean `main` at `194affbeadba1ae08da7f419786640fdb9171b4d`. This record covers the initial documentation and packaging checks and the subsequent independent experiment.

## Starting state and scope

| Phase | Starting status | Evidence inspected |
|---|---|---|
| 0: protocol/setup | Complete, with packaging metadata defects addressed in Phase 8 | Plan, instructions, canonical config, package layout, Phase 0 tests |
| 1: acquisition/audit | Complete | Pinned provenance, raw checksum, validation/ID manifests, duplicate and text audits/figures, Phase 1 tests |
| 2: leakage-safe splitting | Complete | Original-record manifest, 7,000/1,500/1,500 counts, all ten saved leakage checks, Phase 2 tests |
| 3: frozen features | Complete | Normalized cache metadata, 384-d encoder wrapper, 1,543-d schema/matrices, training scaler, Phase 3/normalization tests |
| 4: baselines/control | Complete | Validation-only selection report, preserved baseline checkpoint, Phase 4 tests |
| 5: MLP | Complete | Three checkpoint files, validation report, histories and loss curves, Phase 5 tests |
| 6: evaluation/shuffle | Complete | One completed locked evaluation, metrics/predictions, curves/confusion matrices, mapping checks, Phase 6 tests |
| 7: qualitative analysis | Complete | 10 FP + 10 FN notes/reviews, evidence, descriptive report, preservation manifest, Phase 7 tests |
| 8: documentation/reproducibility | Partial | README needed reorganization; clean-environment verification absent; full fresh reproduction constrained by historical locks |

Recent relevant commits: `194affb` qualitative analysis, `a4d6d98` locked evaluation/shuffle, `dd7ade7` three-seed training, `6cb4895` validation-selected baselines. Existing valid implementations and experimental artifacts were reused. The independent workflow adds workspace and provenance handling, revision pinning, validation-seed selection and review export. The canonical configuration and published experimental artifacts retain their original hashes.

Canonical settings remain: labels 0/1 (Hallucinated positive); split 70/15/15 seed 42; frozen normalized MiniLM 384-d; chunk size/overlap 192/32; four interaction blocks plus seven scalars = 1,543; MLP 1543/256/64/1, ReLU, dropout 0.20/0.10, BCEWithLogitsLoss, AdamW 0.001, weight decay 0.0001, batch 128, maximum 30 epochs, validation-loss patience 5; seeds 101/202/303. The pre-existing, explicitly authorized validation-only cosine-direction search is documented, not changed.

## Changes and dependency inspection

- Reorganized the README around question, findings, methods, exact saved test results, diagnostics, limits and citations; embedded only ROC and loss figures and linked other details.
- Added [stage-by-stage reproduction guidance](REPRODUCING.md), with separate offline tests, earlier-stage regeneration, independent reproduction and preserved-note export.
- Made editable package dependencies derive from `requirements.txt`; added directly imported `joblib` and `threadpoolctl`; corrected the advertised Python floor to >=3.10 and removed the stale Phase 0 package description. No main-environment packages were installed/upgraded.
- Added [observed direct constraints](../constraints-tested.txt), [main-environment version record](environment-observed.json), and three focused documentation tests. A source-root editable installation remains the supported layout; standalone-wheel configuration portability was not changed.

An AST import scan of `src/` and `scripts/` found third-party imports of NumPy, scikit-learn, Matplotlib, PyYAML, sentence-transformers, PyTorch, joblib and threadpoolctl. Existing pandas/transformers/tqdm requirements were retained. Standard-library modules need no extra dependencies. The test suite uses pytest. No unrelated platform-specific dependency was added.

## Exact commands and outcomes

### Existing main environment

```powershell
.venv\Scripts\python.exe -m pip list --format=json
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Smoke commands (run separately from the suite):

```powershell
.venv\Scripts\python.exe -c "from hallucination_detector.config import load_config; from hallucination_detector.mlp import HallucinationMLP; c=load_config(); print(c['primary_features']['total_dim']); print(HallucinationMLP(c['mlp']).__class__.__name__)"
.venv\Scripts\python.exe scripts/run_phase3.py --help
.venv\Scripts\python.exe scripts/run_phase6.py --help
.venv\Scripts\python.exe scripts/run_phase7.py --help
```

Before edits, **183 passed in 57.36s**, with no skips or reported pytest warnings. `pip check` reported no broken requirements. After edits, **186 passed in 81.13s**, including the three new documentation tests, with no skips or reported pytest warnings. The package/config/MLP smoke printed `1543` and `HallucinationMLP`; Phase 3/6/7 `--help` commands exited successfully without data/model downloads or evaluation. The Phase 6 help import emitted a nonfatal Matplotlib font-cache permission warning; the command still succeeded. No main-environment packages were installed or upgraded.

### Initial separate temporary environment

The clean environment is a new venv without `--system-site-packages`; the working `.venv` was not modified. Its temporary path is ignored by Git.

```powershell
.venv\Scripts\python.exe -m venv tmp/phase8/clean-env
tmp/phase8/clean-env/Scripts/python.exe -m pip install -c constraints-tested.txt -e .
```

The first installation failed before project installation while resolving isolated build dependencies (`setuptools>=61.0`): sandbox DNS resolution returned `getaddrinfo failed`. The permitted network retry used:

```powershell
tmp/phase8/clean-env/Scripts/python.exe -m pip install --retries 0 --timeout 15 -c constraints-tested.txt -e .
```

The retry **installed successfully** and built the editable package. Installed package metadata declares all 12 requirements, including joblib/threadpoolctl. `sys.prefix != sys.base_prefix` and `include-system-site-packages = false` confirmed isolation. Clean `pip check` reported no broken requirements. [Resolved clean-environment versions](environment-clean.json) record differing transitive versions separately; direct constraints are explicitly not a transitive lock. The pip upgrade notice was informational; pip was not upgraded. The clean suite completed with **186 passed in 109.09s**, with no skips or reported pytest warnings. The sentence-transformers/pandas/transformers import smoke succeeded at versions **6.0.1 / 3.0.5 / 5.16.1**. Package/config/MLP and Phase 3/6/7 help checks also passed; no data or encoder weights were downloaded. The only installation notices were the initial sandbox DNS failure (resolved by the permitted retry) and an informational pip upgrade notice.

Clean checks ran with downloads disabled for Hugging Face libraries and an ignored writable Matplotlib cache:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:MPLCONFIGDIR=(Join-Path (Get-Location) 'tmp/phase8/mpl-clean')
tmp/phase8/clean-env/Scripts/python.exe -m pip check
tmp/phase8/clean-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
tmp/phase8/clean-env/Scripts/python.exe -c "from hallucination_detector.config import load_config; from hallucination_detector.mlp import HallucinationMLP; c=load_config(); print(c['primary_features']['total_dim']); print(HallucinationMLP(c['mlp']).__class__.__name__)"
tmp/phase8/clean-env/Scripts/python.exe scripts/run_phase3.py --help
tmp/phase8/clean-env/Scripts/python.exe scripts/run_phase6.py --help
tmp/phase8/clean-env/Scripts/python.exe scripts/run_phase7.py --help
$env:MPLCONFIGDIR=(Join-Path (Get-Location) 'tmp/phase8/mpl-clean-import')
tmp/phase8/clean-env/Scripts/python.exe -c "import sentence_transformers, pandas, transformers; print('Library import smoke:', sentence_transformers.__version__, pandas.__version__, transformers.__version__)"
```

The initial package/config/MLP smoke and all three CLI help checks succeeded. Those checks were independent of the subsequent real-data run.

## Repository checks and artifact preservation

Before edits, SHA-256 values were captured for **88 files / 386,815,733 bytes** under `data/`, `results/`, `checkpoints/` and `configs/`, including ignored data, caches, scalers and all four classifier checkpoint files. Final comparison is recorded in [the preservation manifest](phase8_artifact_preservation.json). No experimental command writes to those paths in this phase. A read-only call to `validate_phase2_leakage` checked the existing raw/expanded data and manifests: **all 11 current checks passed**, covering 10,000 sources and 20,000 expanded rows. (The older saved report contains ten checks; the current function additionally checks exact-duplicate groups.) A disposable-copy call to `export_phase7` reproduced selection, reviewed cases, category counts and report **byte for byte** from the preserved notes, without inference or writing into the reference artifacts. These outcomes are included in the preservation manifest.

Checks include local Markdown links, figure/artifact references, script paths, exact inclusion of the saved final comparison table, Git ignore rules, changes for secrets/local absolute paths/temporary files, and `git diff --check`. The checks found no broken local links, missing referenced figures/artifacts/scripts, changed final-table values, newly introduced key-like secrets/local absolute paths, or trailing whitespace. `git check-ignore` confirmed ignored raw/processed data, embedding caches, matrices, scaler, checkpoints, environments, temporary paths, `.env`/sample `.env` and `secrets.yaml`. `git ls-files` found none of the ignored data/cache/checkpoint/environment/temp paths tracked. Historical Phase 7 preservation checksums and Phase 6 primary-output checksums also matched. `git diff --check` passed; Git emitted only its LF-to-CRLF conversion advisory for the rewritten README.

Upstream citation and license checks used the [HaluEval paper](https://aclanthology.org/2023.emnlp-main.397/), [official HaluEval repository](https://github.com/RUCAIBox/HaluEval) (explicit MIT statement and QA/HotpotQA/Wikipedia provenance), [Sentence-BERT paper](https://aclanthology.org/D19-1410/), and [encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0). Fetching the historical pinned LICENSE URL through the web tool failed; the README links the verified upstream repository license and does not claim that the historical license file was independently fetched. No project-code license was invented.

## Independent run

The run used the clean Windows CPython 3.12.0 environment recorded in `environment-clean.json` and the exact stage commands in the [reproduction guide](REPRODUCING.md). Outputs are local under ignored `tmp/phase8-independent/`. The workspace manifest records the source HEAD and SHA-256 of the copied code, configuration, tests, and reference inputs. The existing-target CLI check returned exit 2.

| Stage | Outcome and inputs |
|---|---|
| Raw dataset | Reused the published raw bytes and provenance after SHA-256 verification against pinned revision `207f479c555a68b8d2431631a95083b84b2accbe`. |
| Validation and auditing | Ran validation, duplicate audit, and text-level audit on all 10,000 source records. Zero exclusions. |
| Split and expansion | Ran manifest generation, expansion, and leakage checks. The split manifest matched the published SHA-256, all 11 checks passed, and the three splits contained 14,000 / 3,000 / 3,000 examples. |
| Encoder and features | Reused the cached pretrained model and tokenizer at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The cached safetensors digest matched upstream. Generated fresh embeddings for 10,000 questions, 17,698 answers, 9,936 contexts and 9,994 chunks. Rebuilt the 1,543-column feature matrices, fitted the scaler on training rows, and ran the train-only similarity audit. |
| Baselines and MLP | Fitted the baselines and all three MLP seeds from the generated training features. The validation F1 values were 0.940160, 0.940274, and 0.943610 for seeds 101, 202, and 303. Median validation F1 selected seed 202. |
| Evaluation and shuffle | Ran validation-only smoke and lock preparation, then one final test evaluation and one prespecified context shuffle. The execution record reports 1/1 and complete. Test features reconstructed exactly and all shuffle integrity checks passed. |
| Qualitative export | The 10 FP and 10 FN candidates and their evidence matched the published reviews. Reused notes and reproduced `reviewed_cases.json` byte for byte. In a separate disposable copy with one altered excerpt, export produced 20 review candidates, zero reviewed cases and `fresh_analyst_review_required`. |

The primary metric records, three-seed summary and all shuffled comparisons matched the published values exactly. The comparison table and both prediction CSVs matched byte for byte.

| Approach | Final F1 | Final ROC-AUC | Shuffled F1 | Shuffled ROC-AUC |
|---|---:|---:|---:|---:|
| Cosine | 0.724895 | 0.746504 | 0.038560 | 0.465869 |
| Primary Logistic Regression | 0.947931 | 0.985544 | 0.734623 | 0.949069 |
| Representative MLP (202) | 0.938762 | 0.983063 | 0.758408 | 0.956576 |
| Three-seed MLP mean | 0.938771 | 0.983028 | — | — |
| Answer-only negative control | 0.947333 | 0.974362 | — | — |

The full current suite passed with `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` (**189 passed**). `git diff --check` passed and all 88 published SHA-256 values matched the preservation manifest after the run. The encoder logged a tokenizer length advisory while tokenizing full contexts during chunking. Feature generation completed and the evaluator reconstructed the test features exactly. The host interruption inflated the Phase 3 wall-clock measurement, so that value is unsuitable for runtime estimates.

## Remaining limits and core definition of done

The workflow is verified with checksum-verified cached raw data and a pinned pretrained encoder snapshot. A network-only acquisition on a new machine was outside this verification. The historical encoder revision remains unknown, direct package constraints leave transitive versions open, and the exact match is specific to the tested environment.

| Core definition-of-done criterion (PROJECT_PLAN.md §20) | Status and evidence |
|---|---|
| Official QA provenance documented | Satisfied: pinned source revision, URL, checksum, count and metadata |
| Validation and shortcut auditing reproducible | Satisfied by independent validation and audits on the pinned dataset |
| Original split precedes paired expansion | Satisfied by implementation, saved manifest and tests |
| Leakage assertions pass | Saved all-pass report; synthetic suite checks leakage/determinism |
| Frozen chunked MiniLM used | Satisfied for published artifacts; exact historical revision unavailable |
| Primary 1,543-feature schema | Satisfied by canonical schema, pipeline and tests |
| Three primary approaches compared fairly | Satisfied by saved lock, common test IDs and shared LR/MLP features |
| Answer-only negative control separate | Satisfied in reports and revised README |
| Three MLP seeds evaluated | Satisfied: 101/202/303 checkpoints, histories and test metrics |
| Required metrics and plots complete | Satisfied: saved accuracy/precision/recall/F1/AUC, confusion matrices, ROC/PR, losses |
| Shuffled-context test complete | Satisfied: saved one-shuffle mapping, integrity and fixed-model metrics |
| At least 20 high-confidence errors reviewed | Satisfied: 10 FP + 10 FN, explicit notes/evidence, descriptive limits |
| README methods/results/limits/citations/reproduction | Satisfied by the documented and verified independent workflow |
| Conclusions faithful to evidence | Satisfied: LR wins, near-matching control, high shuffled AUC, benchmark-only interpretation, no significance claim |

## Phase 8 acceptance status

| Acceptance criterion | Outcome |
|---|---|
| New user can reproduce the core experiment | Satisfied on the tested environment with pinned data and encoder inputs |
| README distinguishes primary models and control | Satisfied |
| Shortcut and shuffled-context results discussed | Satisfied |
| Claims remain benchmark-specific | Satisfied |
| No fabricated values or unsupported claims | Saved comparison retained exactly and independently reproduced |
| Demo only after core completion | No demo attempted; explicitly excluded from this phase |
| Required clean-environment verification output | Initial clean install and 186 tests passed. The independent pipeline ran in that environment, and the current full suite passed 189 tests in the main environment. |

## Final changed-file inventory

The initial documentation and packaging changes remain in `README.md`, `pyproject.toml`, `requirements.txt`, `constraints-tested.txt`, `docs/`, and `tests/test_phase8.py`. The independent workflow adds `scripts/init_reproduction.py`, `src/hallucination_detector/reproduction.py`, and `tests/test_reproduction.py`. It updates the Phase 3 and 7 entry points and the encoder, MLP provenance, evaluation and error-analysis modules. The guide and this record include the verified run.

The changes are unstaged on `main` at `194affbeadba1ae08da7f419786640fdb9171b4d`. The verification workspace, environment and disposable mismatch check are under ignored `tmp/`.
