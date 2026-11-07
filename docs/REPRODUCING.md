# Reproduction guide

Start at the published repository root. Section A checks the software and Section B creates an isolated workspace for the experiment. See [verification outcomes](PHASE8_VERIFICATION.md).

## Environment and installation

The observed experiment and verification environment is CPython **3.12.0, Windows AMD64**. `pyproject.toml` requires Python >=3.10 because source annotations use Python 3.10 syntax; other versions/platforms are unverified. The [observed environment](environment-observed.json) records installed versions without local paths. [Direct constraints](../constraints-tested.txt) constrain the direct packages to observed versions, not every transitive dependency. They do not guarantee future index availability or bitwise reproducibility.

In a new checkout, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -c constraints-tested.txt -e .
.venv\Scripts\python.exe -m pip check
```

The editable install reads the single dependency list in `requirements.txt`; installing it separately with `python -m pip install -r requirements.txt` is supported, but the package must also be installed with `python -m pip install -e .` for the early scripts to import it. Installing only requirements is insufficient. Run from the checkout root; a standalone wheel outside the source checkout is not a supported execution layout because canonical `configs/` is repository-relative. No new packaging framework is required.

Package installation needs network access unless the wheels and build dependencies are cached. The initial data step downloads the official pinned HaluEval file from GitHub; the initial encoder step downloads MiniLM weights/tokenizer from Hugging Face unless already cached. CPU is supported. The historical MLP/evaluator used CPU, one PyTorch thread, one BLAS thread, and deterministic algorithms. The Phase 3 CLI chooses CUDA if available and CPU otherwise; its `FrozenEncoder(device="cpu")` API supports explicit CPU execution. No minimum hardware specification or end-to-end runtime measurement is established here.

## A. Offline software verification

After installation, no dataset/model download is needed:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "from hallucination_detector.config import load_config; from hallucination_detector.mlp import HallucinationMLP; c=load_config(); print(c['primary_features']['total_dim']); print(HallucinationMLP(c['mlp']).__class__.__name__)"
.venv\Scripts\python.exe scripts/run_phase3.py --help
.venv\Scripts\python.exe scripts/run_phase6.py --help
.venv\Scripts\python.exe scripts/run_phase7.py --help
```

Tests use small synthetic fixtures, fake encoders, mocked downloads, and temporary directories. Their synthetic evaluation tests do not evaluate the real final-test set. Installation, test, and CLI success do **not** establish full experimental reproduction.

## B. Independent experiment workspace

The initializer copies the current source, scripts, tests, configuration, packaging files, and the published provenance and qualitative reference inputs into a new directory. The workspace manifest records the source HEAD and SHA-256 of every copied input, including uncommitted code. Generated directories start empty. Existing targets are refused. Choose a new name if initialization was interrupted.

The verification used the clean environment from the Phase 8 installation check. Point `$python` at an absolute path to an installed environment before changing directories. Set `PYTHONPATH` to the isolated source so the early scripts use the copied code even if the environment was installed editable in the published repository.

```powershell
$source = (Get-Location).Path
$python = Join-Path $source 'tmp/phase8/clean-env/Scripts/python.exe'
$target = Join-Path $source 'tmp/phase8-independent'
& $python scripts/init_reproduction.py $target
if ($LASTEXITCODE -ne 0) { throw 'Workspace initialization failed' }
$expected = (Get-Content (Join-Path $target 'reference/halueval_qa_provenance.json') -Raw | ConvertFrom-Json).sha256
if ((Get-FileHash data/raw/qa_data.json).Hash.ToLowerInvariant() -ne $expected) { throw 'Raw dataset hash mismatch' }
Copy-Item -LiteralPath data/raw/qa_data.json -Destination (Join-Path $target 'data/raw/qa_data.json')
Copy-Item -LiteralPath (Join-Path $target 'reference/halueval_qa_provenance.json') -Destination (Join-Path $target 'data/metadata/halueval_qa_provenance.json')
Set-Location -LiteralPath $target
$env:PYTHONPATH = Join-Path $target 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:MPLCONFIGDIR = Join-Path $target 'mpl-cache'
```

The copy commands reuse the original checksum-verified raw bytes and their provenance. To acquire the data on a new machine, omit the raw/provenance copy steps and run `& $python scripts/acquire_halueval.py` in the isolated directory with network access. Compare the downloaded file against the published SHA-256 below before proceeding. The source repository's full synthetic test suite also checks its published documentation. Run that suite from the source repository as in Section A.

### 1. Acquisition, validation, IDs and audit

```powershell
& $python scripts/validate_halueval.py
& $python scripts/duplicate_audit.py
& $python scripts/audit_halueval.py
```

The acquisition script downloads `data/qa_data.json` from official revision `207f479c555a68b8d2431631a95083b84b2accbe`. The file is NDJSON despite its `.json` suffix. Expected SHA-256 is **`89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`**, size **6,164,420 bytes**, count **10,000**, with fields `question`, `knowledge`, `right_answer`, `hallucinated_answer`. The tracked [provenance](../data/metadata/halueval_qa_provenance.json) records the original download date, URL, revision, checksum and license.

Verify the raw checksum before continuing:

```powershell
(Get-FileHash data/raw/qa_data.json -Algorithm SHA256).Hash.ToLowerInvariant()
```

Acquisition refuses to replace existing raw bytes without its explicit force option; the wrapper still writes provenance for an existing file. It records a new timestamp when run again. Thus provenance JSON need not be byte-identical even if raw data are identical. Validation records exclusions and deterministic source IDs; the published run has 10,000 valid records and zero exclusions. Audit outputs are text-level shortcut distributions and hash/ID-based duplicate summaries, not new model selection.

### 2. Split originals, then expand and check leakage

```powershell
& $python scripts/create_split_manifest.py
& $python scripts/expand_split_records.py
& $python scripts/validate_split_leakage.py
```

The split command groups duplicate normalized contexts and assigns original records with seed 42 and requested 70/15/15 proportions. It saves `data/splits/halueval_original_split_manifest.json` before expansion. The manifest has IDs, original indices, groups and split assignments, without raw text. The published manifest SHA-256 is **`cec404672fb3a9994a6f6413793a774fbdb46de718b5294ebdc4158b9c13ce6b`**; compare it before generating dependent features. Split metadata includes a creation timestamp and may differ even when the manifest matches.

Expansion creates exactly two examples per source with `0 = Grounded`, `1 = Hallucinated`. The leakage command must report `all_passed: true`: no source/context group crosses splits, pairs stay together, every valid source appears exactly once, each split is balanced, and example IDs and manifests are deterministic. See [saved leakage assertions](../data/splits/halueval_split_leakage_report.json).

### 3. Frozen embeddings, features and training-only scaler

```powershell
& $python scripts/run_phase3.py --encoder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
& $python scripts/audit_train_similarity.py
```

The pipeline loads normalized questions/answers/contexts, chunks contexts at 192 tokens with 32 overlap, caches frozen MiniLM embeddings, constructs all 1,543 primary features, fits scaling on training rows only, and transforms all splits. It generates test features without model/threshold selection. The semantic audit reads training features/labels only.

Compatible caches are reused; incomplete or incompatible caches fail clearly. Metadata binds dataset/manifest checksums, preprocessing **2.0.0**, feature schema **1.0.0**, encoder identity/revision, dimension, normalization, and chunking. The existing `--force-recompute` option replaces caches; it is not needed for offline checks and must not be used against preserved experiment caches.

The independent CLI requires a full Hugging Face commit SHA and passes it to the model loader. The verification reused the cached weights and tokenizer at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Set the Hugging Face offline flags only when this snapshot is available locally. The cached `model.safetensors` SHA-256 matched the upstream digest `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`. The published encoder revision remains unknown. [Pinned upstream weights](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors)

### 4. Baselines and exactly three MLP seeds

```powershell
& $python scripts/run_phase4.py
& $python scripts/run_phase5.py
```

Phase 4 fits primary LR and the answer-only control on training data, searches validation-only regularization/thresholds and both authorized cosine directions, and saves `results/metrics/phase4_validation.json` plus `checkpoints/phase4/baselines.joblib`. Both primary LR and MLP receive the same training-scaled 1,543 features. The answer-only control fits its own training scaler. These training commands can overwrite their reports/checkpoints; keep them confined to the independent regeneration checkout.

Phase 5 verifies Phase 4 provenance and scaler/feature identities, trains **101/202/303**, selects checkpoints by validation loss, selects thresholds by validation F1, and picks the median-validation-F1 representative. It writes `phase5_validation.json`, all seed histories, `phase5_losses.png`, and `mlp_seed_{101,202,303}.pt`. Neither training workflow opens test features/labels. Checkpoints are ignored local files; only trusted baseline joblib files should be loaded.

### Local artifact dependency order

| Generated locally; ignored by Git | Producer | Needed by |
|---|---|---|
| `data/raw/qa_data.json` | Acquisition | Validation, audit, expansion |
| `data/processed/halueval_expanded_{all,train,validation,test}.jsonl` | Split then expansion | Features; answer-only training; evaluation/qualitative joins |
| `data/embedding_cache/` including IDs, arrays, offsets and metadata | Phase 3 | Feature reuse and frozen shuffled-context recomputation |
| `data/features/{train,validation,test}/` including raw/scaled arrays, labels, IDs, metadata; `data/features/schema.json` | Phase 3 | Baselines, MLP, evaluation |
| `data/scaler/scaler.npz` | Phase 3 training rows | Every subsequent primary-model transformation |
| `checkpoints/phase4/baselines.joblib` | Phase 4 | Preparation and fixed evaluation |
| `checkpoints/phase5/mlp_seed_{101,202,303}.pt` | Phase 5 | Preparation and fixed evaluation |

Tracked metadata, split manifests, validation reports, predictions, figures, and analyst notes are reference evidence; they do not substitute for missing ignored artifacts. Preserve coherent bundles: mixing regenerated features with historical checkpoints/reports will correctly fail checksum or compatibility checks.

## C. Lock and evaluate the independent run

Run these in the workspace after Phase 5 completes. Check each command succeeds before proceeding.

```powershell
& $python scripts/run_phase6.py --prepare
& $python scripts/run_phase6.py --evaluate
```

Preparation checks validation-only inference and locks the independent run's checkpoints, thresholds, representative seed, and input hashes. The representative follows median validation F1 with the existing smaller-seed tie rule. Evaluation writes its attempt record, saves primary results, and performs one prespecified context shuffle. The existing locks refuse a second preparation or evaluation attempt. Keep the published result tree and locks in the source repository.

## D. Qualitative export

```powershell
& $python scripts/run_phase7.py
```

The exporter joins the locked representative model's original predictions to the processed examples and selects up to 10 high-confidence errors of each type. Independent output contains `selection.json` and `review_candidates.json` under `results/error_analysis/phase7/`. It emits `reviewed_cases.json` only if all selected IDs, case text, decisions and evidence match the published reviews. Otherwise the manifest flags `fresh_analyst_review_required` and records zero reviewed cases. Keep review candidates local because they contain raw benchmark text. Independent export refuses to overwrite an existing review directory.

## Protection and interpretation

Keep raw/processed text, embeddings, matrices, scalers, environments and checkpoints ignored. Preserve tracked provenance, manifests, metrics, plots and qualitative evidence. [The verification report](PHASE8_VERIFICATION.md) records artifact-preservation checks. Never use final-test or shuffled-context results to choose features, thresholds, regularization, epochs, seeds, or extensions. Three-seed sample SD measures only observed training-run variation; the benchmark results do not establish general-purpose grounding reliability.
