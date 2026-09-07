# PROJECT PLAN: Neural Hallucination Detection for Large Language Model Responses

## 1. Project Overview

This project investigates whether a lightweight neural classifier can distinguish grounded answers from hallucinated answers when given:

- A question
- A reference context
- A candidate answer

The project uses the HaluEval question-answering subset and is designed as a small, reproducible research experiment rather than a production hallucination detector or application-focused demo.

The central comparison is between:

1. A cosine-similarity threshold baseline
2. A Logistic Regression baseline
3. A lightweight PyTorch multilayer perceptron (MLP)

The sentence-embedding model will remain frozen. No large language model or transformer will be fine-tuned.

---

## 2. Research Question

On paired HaluEval-QA examples, does a lightweight nonlinear MLP trained on frozen sentence-embedding interaction features detect hallucinated answers more effectively than an answer-context cosine threshold and Logistic Regression using comparable features?

### Primary hypothesis

The MLP will achieve higher hallucinated-class F1 and ROC-AUC than the cosine-similarity baseline because it can learn nonlinear relationships between the question, candidate answer, and supporting context.

### Secondary hypothesis

The MLP may outperform Logistic Regression if the relevant embedding interactions are not linearly separable.

### Null or negative result

If Logistic Regression matches or outperforms the MLP, the experiment is still considered successful.

Such a result would suggest that nonlinear classification adds limited value for this feature representation or that benchmark artifacts dominate the task.

No minimum performance improvement is required for project completion.

---

## 3. Scope and Claims

### In scope

- HaluEval QA acquisition and validation
- Leakage-safe paired dataset construction
- Exploratory dataset and shortcut analysis
- Frozen sentence embeddings
- Context chunking
- Engineered embedding-interaction features
- Cosine-similarity baseline
- Logistic Regression baseline
- Lightweight PyTorch MLP
- Reproducible training and evaluation
- Quantitative model comparison
- Qualitative error analysis
- Clear research documentation
- Optional minimal local demo

### Out of scope

- Fine-tuning sentence transformers or large language models
- Paid APIs
- External LLM APIs
- Generating new hallucinated answers
- Multi-dataset generalization experiments
- Production deployment
- Claim-level hallucination localization
- Retrieval-augmented generation
- Complex hyperparameter optimization
- Formal human evaluation
- Claims of general real-world hallucination detection

### Interpretation boundary

The final conclusions must be limited to the paired HaluEval-QA benchmark.

The project must not claim that strong benchmark performance proves reliable hallucination detection for arbitrary LLMs, domains, answer styles, or real-world class distributions.

Because every primary method uses frozen neural sentence embeddings, the comparison should be described as:

- Threshold-based classification over frozen embeddings
- Linear classification over frozen embedding features
- Nonlinear neural classification over frozen embedding features

It should not be described as a completely neural system versus completely non-neural systems.

---

## 4. Dataset

### 4.1 Source

Use the official HaluEval QA data.

Primary references:

- Repository: https://github.com/RUCAIBox/HaluEval
- Paper: https://aclanthology.org/2023.emnlp-main.397/

Each original QA record is expected to contain:

- `question`
- `knowledge`
- `right_answer`
- `hallucinated_answer`

The repository must record:

- Dataset source URL
- Download date
- Dataset file checksum
- Number of raw records
- Number of valid records
- Number of excluded records
- Reasons for exclusions
- License information
- Citation information

Raw data must not be silently modified.

### 4.2 Labels

Use the following fixed label convention:

- `0 = Grounded`
- `1 = Hallucinated`

The hallucinated class is the positive class for precision, recall, and F1 reporting.

### 4.3 Paired expansion

Every valid original record produces exactly two classification examples:

1. `question + knowledge + right_answer -> Grounded`
2. `question + knowledge + hallucinated_answer -> Hallucinated`

Each expanded row must retain the `source_id` of its original record.

### 4.4 Stable source identifiers

If the source dataset does not provide a reliable unique identifier, create a deterministic identifier from a stable hash of the original record fields.

The identifier must be created before dataset expansion and must remain reproducible across runs.

---

## 5. Leakage Prevention and Dataset Splitting

### 5.1 Required processing order

The dataset must be split at the original-record level before paired expansion.

Required order:

1. Load original HaluEval records.
2. Validate records using documented rules.
3. Assign stable `source_id` values.
4. Audit exact duplicates.
5. Split original records into training, validation, and test groups.
6. Expand the records within each split into grounded and hallucinated rows.

The expanded rows must never be randomly split independently.

### 5.2 Split proportions

Use:

- 70% training records
- 15% validation records
- 15% test records

Use one fixed and documented split seed.

Because every source record contributes one example from each class, every split should be balanced.

Records must be validated before splitting so that later removal does not disturb this balance.

### 5.3 Duplicate handling

Before splitting, audit:

- Exact duplicate records
- Duplicate normalized questions
- Duplicate contexts
- Duplicate question-context combinations
- Identical grounded and hallucinated answers
- Empty or malformed fields

Exact duplicate records should be removed or assigned to the same split group.

Duplicate question-context records should be grouped where practical so closely related copies do not cross splits.

Near-duplicate detection is optional and must not delay the core project.

### 5.4 Required leakage assertions

Automated checks must verify that:

- No `source_id` occurs in more than one split.
- Both expanded rows from every source record occur in the same split.
- No known exact-duplicate group crosses splits.
- Every valid source record produces exactly two expanded rows.
- Every split is balanced.
- Saved split manifests reproduce the same assignments when rerun.

---

## 6. Dataset and Shortcut Audit

HaluEval’s correct and hallucinated answers come from different creation processes. A classifier may therefore learn answer style or length instead of grounding.

Before model training, report class-wise distributions for:

- Answer token count
- Answer character count
- Answer sentence count
- Numeric-token frequency
- Answer-context cosine similarity
- Answer-context lexical coverage
- Exact answer occurrence in the context
- Missing or malformed values

At minimum, produce:

- A summary statistics table by class
- An answer-length distribution visualization
- An answer-context similarity distribution visualization
- A written interpretation of major class differences

These variables must be treated as possible confounders rather than automatically included in the primary model.

Standalone answer-length and stylistic features must not be included in the primary feature vector.

---

## 7. Text and Embedding Pipeline

### 7.1 Encoder

Use:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Frozen encoder
- L2-normalized embeddings

Record the installed package version and, when available, the model revision.

CPU execution must be supported. GPU execution may be used when available.

### 7.2 Text preprocessing

Preprocessing should remain minimal:

- Convert valid fields to strings.
- Normalize repeated whitespace.
- Strip leading and trailing whitespace.
- Preserve meaningful case, punctuation, numbers, and negation.
- Do not aggressively remove stopwords.
- Do not stem or lemmatize text used by the sentence encoder.

Any separate token normalization used for lexical coverage must not alter the original encoder input.

### 7.3 Context chunking

Long reference contexts must be chunked before embedding.

Core configuration:

- Maximum chunk size: 192 model tokens
- Chunk overlap: 32 model tokens
- Tokenizer-aware boundaries where practical

For each example:

- Embed the question as `q`.
- Embed the candidate answer as `a`.
- Embed all context chunks as `c1, ..., cn`.
- Select `ca`, the context chunk with the highest cosine similarity to the candidate answer.

Also calculate the maximum question-context similarity as a scalar feature.

Full question-context interaction vectors are optional and are not part of the core primary architecture.

### 7.4 Embedding caching

Cache embeddings to avoid repeated encoding.

Cache metadata must include:

- Encoder name
- Encoder revision, if available
- Embedding dimension
- Normalization setting
- Chunk size
- Chunk overlap
- Dataset checksum
- Preprocessing version
- Feature-pipeline version

A cache must not be reused when incompatible metadata changes.

---

## 8. Primary Feature Representation

The primary representation contains four 384-dimensional interaction blocks.

Let:

- `q` be the question embedding.
- `a` be the candidate-answer embedding.
- `ca` be the answer-selected context-chunk embedding.
- `⊙` denote element-wise multiplication.

Use the following vector features:

1. `|a - ca|`
2. `a ⊙ ca`
3. `|q - a|`
4. `q ⊙ a`

These contribute:

`4 × 384 = 1,536 features`

### Core scalar features

Add the following seven scalar features:

1. Maximum answer-context-chunk cosine similarity
2. Mean of the top three answer-context-chunk similarities
3. Question-answer cosine similarity
4. Maximum question-context-chunk cosine similarity
5. Answer content-token coverage in the context
6. Answer numeric-token coverage in the context
7. Indicator that the answer contains at least one numeric token

Total primary feature dimension:

`1,536 + 7 = 1,543 features`

If a context has fewer than three chunks, average all available chunks for the top-three similarity feature.

For an answer without numeric tokens, use:

- Numeric coverage value: `0`
- Numeric-presence indicator: `0`

### Feature scaling

Fit the feature scaler using training data only.

Apply the fitted training scaler unchanged to:

- Validation data
- Test data
- Error-analysis examples
- Demo input

Logistic Regression and the MLP must use the same primary feature representation.

---

## 9. Models

### 9.1 Cosine-similarity baseline

Use maximum answer-context-chunk cosine similarity as the baseline score.

Expected direction:

- Higher similarity suggests grounded.
- Lower similarity suggests hallucinated.

Requirements:

- Select the classification threshold using validation data only.
- Optimize the threshold for hallucinated-class F1.
- Use a continuous score with the direction reversed where necessary for hallucination ROC-AUC.
- Do not select or adjust the threshold using test labels.

### 9.2 Logistic Regression baseline

Train Logistic Regression on the 1,543-dimensional primary feature vector.

Core configuration:

- L2 regularization
- Sufficient maximum iterations for convergence
- Fixed random state where applicable
- No class weights unless an unexpected imbalance is discovered and documented

Use the following small validation-only search:

- `C = 0.01`
- `C = 0.1`
- `C = 1`
- `C = 10`

Select the configuration using validation F1 for the hallucinated class.

### 9.3 Neural MLP

Use the following lightweight PyTorch architecture:

- Input: 1,543 features
- Linear layer: 1,543 -> 256
- ReLU
- Dropout: 0.20
- Linear layer: 256 -> 64
- ReLU
- Dropout: 0.10
- Linear layer: 64 -> 1

Training configuration:

- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW
- Initial learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 128
- Maximum epochs: 30
- Early-stopping patience: 5
- Early-stopping criterion: validation loss
- Best checkpoint: lowest validation loss
- Classification threshold: selected on validation data for hallucinated-class F1

Do not add architectural complexity unless the locked model fails to train correctly.

### 9.4 Random seeds

Train the MLP using three documented random seeds.

Report:

- Mean test metrics across the three runs
- Standard deviation across the three runs

For outputs requiring one representative model, including a confusion matrix and qualitative error analysis, select the run using validation performance only.

Prefer the run with median validation F1.

Never select the representative run using test performance.

If runtime is genuinely negligible, more seeds may be added as an optional extension.

---

## 10. Training Protocol

The training pipeline must:

- Set Python, NumPy, and PyTorch seeds.
- Record package and platform information.
- Load previously saved split manifests.
- Fit preprocessing components on training data only.
- Save the model configuration.
- Save the fitted feature scaler.
- Save model checkpoints.
- Save epoch-level training and validation losses.
- Record the selected validation threshold.
- Avoid inspecting test metrics during model development.

The project should use a small, predefined configuration rather than a large hyperparameter search.

The pretrained sentence encoder must remain frozen throughout the project.

---

## 11. Evaluation Protocol

### 11.1 Primary metric

The primary metric is:

- F1 score for the hallucinated class

### 11.2 Required secondary metrics

Report:

- Accuracy
- Precision for the hallucinated class
- Recall for the hallucinated class
- F1 for the hallucinated class
- ROC-AUC
- Confusion matrix

Use continuous model scores, not hard predictions, to calculate ROC-AUC.

### 11.3 Required visualizations

Produce:

- Model-comparison results table
- Confusion matrix for each primary approach
- ROC curve comparing all three approaches
- Precision-recall curve comparing all three approaches
- MLP training and validation loss curve

If time becomes constrained, prioritize:

1. Results table
2. Confusion matrices
3. ROC curve
4. Precision-recall curve
5. Training-history plot

### 11.4 Test-set policy

The test set is locked during development.

Use validation data for:

- Cosine threshold selection
- Logistic Regression regularization selection
- MLP early stopping
- MLP threshold selection
- Representative MLP seed selection
- Any optional architecture decisions

Run final test evaluation only after the feature pipeline and model-selection rules are fixed.

If a genuine implementation defect is found after test evaluation, document the defect and the reason for rerunning evaluation rather than silently replacing the result.

### 11.5 Comparison table

The README must contain a table with at least:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Cosine threshold | TBD | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Neural MLP | TBD | TBD | TBD | TBD | TBD |

For the MLP, report mean plus or minus standard deviation across three seeds where appropriate.

### 11.6 Outcome-neutral reporting

The final report must not imply that the MLP is superior unless the measured results support that conclusion.

The discussion should distinguish between:

- Numerical improvement
- Consistency across seeds
- Practical significance
- Possible benchmark shortcuts
- Generalization limitations

---

## 12. Qualitative Error Analysis

Analyze high-confidence MLP errors after final evaluation.

Review at least:

- 10 high-confidence false positives
- 10 high-confidence false negatives

A false positive means a grounded answer predicted as hallucinated.

A false negative means a hallucinated answer predicted as grounded.

For every reviewed example, record:

- `source_id`
- Question
- Relevant context excerpt
- Candidate answer
- True label
- Predicted label
- Hallucination score
- Confidence or distance from the classification threshold
- Assigned failure category
- Short analyst note

Use the following initial categories:

- Entity substitution
- Numerical hallucination
- Contradiction
- High lexical overlap but incorrect meaning
- Plausible but unsupported information
- Partial correctness
- Context ambiguity
- Evidence missed because of context selection
- Dataset or label anomaly
- Other or unclear

Categories may be multi-label.

The error analysis is descriptive. Do not present category frequencies from this small selected sample as population-level estimates.

---

## 13. Reproducibility Requirements

The completed repository must allow another user to reproduce:

1. Dataset acquisition
2. Dataset validation
3. Leakage-safe splitting
4. Paired expansion
5. Embedding generation
6. Feature extraction
7. Baseline training
8. MLP training
9. Final evaluation
10. Result figures and tables

Required reproducibility artifacts:

- Dependency specification
- Fixed configuration file
- Random seeds
- Split manifest
- Dataset checksum
- Cached-feature metadata
- Model configuration
- Saved thresholds
- Saved evaluation results
- Clear execution instructions

Generated artifacts must be separated from source code.

Large caches and model checkpoints may be excluded from Git if reproducible generation instructions are provided.

---

## 14. Recommended Repository Structure

    neural-hallucination-detector/
    ├── app/
    ├── configs/
    │   └── default.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── splits/
    ├── notebooks/
    │   └── 01_data_audit.ipynb
    ├── results/
    │   ├── figures/
    │   ├── metrics/
    │   └── error_analysis/
    ├── scripts/
    ├── src/
    │   └── hallucination_detector/
    │       ├── data/
    │       ├── features/
    │       ├── models/
    │       ├── training/
    │       └── evaluation/
    ├── tests/
    ├── AGENTS.md
    ├── PROJECT_PLAN.md
    ├── README.md
    ├── requirements.txt
    └── .gitignore

The exact number of modules should remain modest.

Avoid unnecessary abstraction layers or files containing only a few lines.

A single exploratory notebook is sufficient. The reproducible pipeline must live outside the notebook.

---

## 15. Project Phases and Acceptance Criteria

### Phase 0 — Protocol and repository setup

Tasks:

- Create the repository structure.
- Record the research question and hypotheses.
- Create the default experiment configuration.
- Define seeds, labels, metrics, and split proportions.
- Create `AGENTS.md`.
- Establish dependency and artifact-management rules.

Acceptance criteria:

- The project installs in a fresh environment.
- The experiment configuration has one source of truth.
- `Hallucinated = 1` is documented.
- Test-set restrictions are documented.
- `AGENTS.md` instructs coding agents to follow this plan.
- No model implementation begins before the protocol is fixed.

### Phase 1 — Dataset acquisition and audit

Tasks:

- Acquire HaluEval QA reproducibly.
- Validate the schema.
- Assign stable source IDs.
- Identify invalid and duplicate records.
- Produce the shortcut-risk audit.

Acceptance criteria:

- Dataset source and checksum are recorded.
- Raw, valid, excluded, and duplicate counts are reported.
- Exclusions have explicit reasons.
- Answer-length and semantic-overlap distributions are compared by class.
- Potential confounders are discussed before training.

### Phase 2 — Leakage-safe splitting and expansion

Tasks:

- Group exact duplicates where necessary.
- Split original records 70/15/15.
- Save the split manifest.
- Expand records into paired labeled rows.
- Add automated leakage tests.

Acceptance criteria:

- No source ID crosses splits.
- No known exact-duplicate group crosses splits.
- Both answers from a source record remain together.
- Every retained source produces exactly two rows.
- Every split is balanced.
- Repeated execution with the same seed reproduces the split.

### Phase 3 — Embedding and feature pipeline

Tasks:

- Implement minimal text normalization.
- Add tokenizer-aware context chunking.
- Produce frozen question, answer, and chunk embeddings.
- Select the answer-relevant context chunk.
- Construct the 1,543-dimensional primary feature vector.
- Cache embeddings and feature metadata.
- Fit scaling using training data only.

Acceptance criteria:

- Feature dimensions are validated automatically.
- Long contexts generate multiple chunks.
- Empty and short contexts are handled safely.
- Cache compatibility is checked before reuse.
- Validation and test data do not influence fitted preprocessing.
- Repeated feature generation is deterministic.

### Phase 4 — Baselines

Tasks:

- Train and validate the cosine-threshold baseline.
- Train and validate Logistic Regression.
- Save validation decisions and configurations.
- Generate preliminary validation metrics.

Acceptance criteria:

- The cosine threshold is selected using validation data only.
- Logistic Regression uses the shared primary features.
- Regularization selection uses validation data only.
- ROC-AUC uses continuous scores.
- Saved predictions can be traced to source IDs.

### Phase 5 — MLP training

Tasks:

- Implement the locked MLP architecture.
- Add deterministic seeding.
- Add early stopping and checkpointing.
- Train three seeds.
- Save training histories and validation thresholds.

Acceptance criteria:

- The sentence encoder remains frozen.
- The input dimension matches the feature contract.
- Best checkpoints are selected using validation loss.
- Classification thresholds are selected using validation data.
- Training does not rely on test metrics.
- All three seeds produce complete and traceable run artifacts.

### Phase 6 — Final evaluation and comparison

Tasks:

- Run final test evaluation.
- Calculate all required metrics.
- Aggregate the MLP seed results.
- Generate comparison plots and confusion matrices.
- Create machine-readable and README-ready result tables.

Acceptance criteria:

- Every model is evaluated on the identical test set.
- Positive-class definitions are consistent.
- All required metrics are reported.
- MLP variability across seeds is reported.
- Representative plots use a validation-selected seed.
- Conclusions match the evidence, including a possible negative result.

### Phase 7 — Qualitative error analysis

Tasks:

- Identify high-confidence false positives and false negatives.
- Review at least 20 total errors.
- Assign predefined failure categories.
- Summarize recurring limitations.

Acceptance criteria:

- At least 10 false positives and 10 false negatives are reviewed when available.
- Every analysis row contains evidence and a written rationale.
- Label anomalies are distinguished from model errors.
- Findings are described as qualitative and descriptive.
- The README contains representative examples or a concise findings summary.

### Phase 8 — Documentation and reproducibility

Tasks:

- Complete the README.
- Add dataset and model citations.
- Document commands and expected outputs.
- Verify the pipeline from a clean environment.
- Remove temporary and generated files from version control.
- Add the optional demo only if time remains.

Acceptance criteria:

- A new user can understand the research question and reproduce the experiment.
- The README includes the three-model comparison table.
- Key plots are visible in the README.
- Limitations and benchmark-specific claims are explicit.
- Configuration, seeds, and artifact locations are documented.
- Core work is complete before optional features are attempted.

---

## 16. Optional Extensions

The following are optional and must not delay the core project.

### Extended question-context features

Add:

- `|q - cq|`
- `q ⊙ cq`

Here, `cq` is the context chunk most similar to the question.

Treat this as a documented feature ablation rather than silently changing the primary feature specification.

### Cluster-bootstrap confidence intervals

Calculate confidence intervals by resampling original source records while keeping paired rows together.

### Additional MLP seeds

Increase from three seeds to five only if runtime is negligible.

### Shortcut negative-control classifier

Train a small diagnostic classifier using only answer-level length and style features.

Use it to estimate how much label information is available without the reference context.

### Shuffled-context stress test

Replace each test context with an unrelated context and measure how much model performance changes.

### Feature ablations

Possible ablations include:

- Scalar features only
- Embedding-interaction features only
- Removing question-answer interactions
- Removing lexical coverage
- Adding question-context interaction blocks

### Calibration analysis

Possible additions include:

- Reliability diagram
- Brier score
- Expected calibration error
- Validation-fitted temperature scaling

Do not describe sigmoid output as calibrated unless calibration is explicitly performed and evaluated.

### Paired metrics

Possible additions include:

- Pairwise ranking accuracy
- Pair consistency accuracy

### Minimal demo

Use Streamlit or Gradio with:

Inputs:

- Question
- Reference context
- Candidate answer

Outputs:

- Grounded or Hallucinated
- Model score

The demo must reuse the exact saved:

- Encoder
- Text preprocessing
- Context chunking configuration
- Feature implementation
- Scaler
- MLP checkpoint
- Classification threshold

The interface must state that its output is an experimental, benchmark-trained prediction and not a factual guarantee.

---

## 17. Six-Day Schedule

### Day 1 — Protocol, setup, and dataset integrity

Complete:

- Phase 0
- Phase 1
- Phase 2

End-of-day target:

- Validated dataset
- Saved leakage-safe split manifest
- Paired expanded splits
- Dataset audit
- Passing leakage tests

### Day 2 — Embeddings and features

Complete:

- Phase 3

End-of-day target:

- Working frozen embedding pipeline
- Context chunking
- Cached embeddings
- Validated 1,543-dimensional features
- Training-only scaler

### Day 3 — Baselines and initial MLP

Complete:

- Phase 4
- Initial Phase 5 training

End-of-day target:

- Validated cosine baseline
- Validated Logistic Regression baseline
- First complete MLP run
- Saved training artifacts

### Day 4 — Repeated training and locked evaluation

Complete:

- Remaining Phase 5
- Phase 6

End-of-day target:

- Three MLP seed runs
- Final test metrics
- Comparison table
- Confusion matrices
- ROC curve
- Precision-recall curve

### Day 5 — Error analysis and research narrative

Complete:

- Phase 7
- Most of Phase 8

End-of-day target:

- Categorized error-analysis table
- Written findings
- README methodology
- README results and limitations
- Citations
- Reproduction instructions

### Day 6 — Verification, polish, and optional work

Priority order:

1. Clean-environment reproducibility check
2. Fix tests and documentation
3. Validate figures and saved artifacts
4. Improve README presentation
5. Attempt one high-value optional extension
6. Add a minimal demo only if all core work is complete

The demo is the first major component to remove if the schedule slips.

---

## 18. Definition of Done

The core project is complete when:

- HaluEval QA is acquired and documented reproducibly.
- Original records are split before paired expansion.
- Leakage assertions pass.
- Dataset shortcut risks are audited.
- Frozen MiniLM embeddings and chunked contexts are used.
- The primary feature vector follows the locked 1,543-feature specification.
- Cosine similarity, Logistic Regression, and MLP are evaluated fairly.
- Three MLP-seed runs are completed.
- All required test metrics and confusion matrices are available.
- At least 20 high-confidence errors are qualitatively analyzed.
- The README contains results, methodology, limitations, citations, and reproduction instructions.
- Conclusions remain faithful to the observed evidence.

The project is not considered unsuccessful if the MLP fails to outperform Logistic Regression.

Experimental validity, transparent analysis, focused scope, and reproducibility take priority over obtaining a preferred result.

---

## 19. Instructions for GitHub Copilot Agent

The companion `AGENTS.md` must treat this document as the methodological source of truth.

GitHub Copilot Agent must:

- Read `PROJECT_PLAN.md` completely before making changes.
- Work on only the phase named in the current prompt.
- Stop after completing the requested phase.
- Never automatically begin the next phase.
- Preserve the locked label convention.
- Preserve the split-before-expansion requirement.
- Never fit transformations using validation or test data.
- Never use test results for feature, model, checkpoint, threshold, or seed selection.
- Never silently change the encoder, context chunking, feature contract, MLP architecture, primary metric, or evaluation protocol.
- Keep optional extensions separate from the core requirements.
- Add or update tests for every phase.
- Keep the implementation small and appropriate for a six-day project.
- Avoid unnecessary abstractions, services, dependencies, or infrastructure.
- Report all assumptions and unresolved issues.

At the end of every phase, Copilot Agent must report:

1. Files created or changed
2. Important implementation decisions
3. Commands and tests run
4. Test outcomes
5. Acceptance criteria satisfied
6. Assumptions
7. Unresolved issues
8. Recommended next phase

If Copilot believes a locked methodological choice is defective or technically impossible, it must stop and report:

1. The affected requirement
2. Evidence of the problem
3. The smallest proposed change
4. The methodological consequences

Copilot must wait for approval before applying such a change.
