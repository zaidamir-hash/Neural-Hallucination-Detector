# PROJECT_PLAN.md

# Neural Hallucination Detection for Large Language Model Responses

## 1. Project Status and Resumption Rule

This is an existing project previously implemented in part by GitHub Copilot Agent.

It is not a fresh repository.

Before continuing implementation, any new coding agent must:

1. Read this document completely.
2. Read `AGENTS.md` completely.
3. Inspect the existing repository.
4. Examine Git history and working-tree status.
5. Identify existing code, tests, configurations, data artifacts, and results.
6. Determine which project phases are complete, partial, not started, or inconsistent.
7. Report its findings before modifying files.
8. Preserve valid existing work.
9. Continue from the first incomplete phase only after user confirmation.

File presence alone does not prove that a phase is complete. Completion requires evidence that the applicable acceptance criteria are satisfied.

Do not rebuild completed phases unnecessarily.

---

## 2. Project Overview

This project investigates whether a lightweight neural classifier can distinguish grounded answers from hallucinated answers when given:

- A question
- A reference context
- A candidate answer

The project uses the HaluEval question-answering subset and is framed as a small, reproducible research experiment.

The main comparison is between:

1. An answer-context cosine-similarity threshold
2. Logistic Regression over engineered frozen-embedding features
3. A lightweight PyTorch multilayer perceptron over the same primary representation

The project also includes:

- A style/length-only negative-control classifier
- A shuffled-context sanity test
- A qualitative analysis of high-confidence errors

The sentence encoder remains frozen. No transformer or large language model is fine-tuned.

---

## 3. Research Question

On paired HaluEval-QA examples, does a lightweight nonlinear MLP trained on frozen sentence-embedding interaction features detect hallucinated answers more effectively than an answer-context cosine threshold and Logistic Regression using the same primary feature representation?

### Primary hypothesis

The MLP may achieve higher hallucinated-class F1 and ROC-AUC than the cosine baseline because it can learn nonlinear relationships among answer-context and question-answer interactions.

### Secondary hypothesis

The MLP may outperform Logistic Regression if the engineered interaction representation is not linearly separable.

### Negative-result policy

The project remains successful if Logistic Regression matches or outperforms the MLP.

A negative result may indicate that:

- The representation is largely linearly separable.
- The MLP adds unnecessary complexity.
- Benchmark artifacts dominate the task.
- The frozen embedding representation does not capture the required factual distinctions.

No minimum performance improvement is required for project completion.

---

## 4. Claims and Interpretation Boundary

Final conclusions must be limited to the paired HaluEval-QA benchmark.

The project must not claim that benchmark performance establishes reliable hallucination detection for:

- Arbitrary LLM responses
- Unseen domains
- Different answer-generation styles
- Real-world class distributions
- Production deployments

All three primary approaches use representations produced by a pretrained neural sentence encoder.

The comparison must therefore be described as:

- A threshold rule over frozen neural embeddings
- A linear classifier over frozen neural features
- A nonlinear neural classifier over frozen neural features

It must not be described as an entirely neural method versus entirely non-neural methods.

---

## 5. Core Requirements

The following are required for project completion:

- Official HaluEval QA acquisition with provenance
- Dataset checksum and pinned source revision
- Original-record validation
- Stable deterministic source IDs
- Duplicate and shortcut-artifact audit
- Original-record splitting before paired expansion
- Duplicate-context-aware train/validation/test grouping
- No source or duplicate-context group crossing splits
- Frozen `sentence-transformers/all-MiniLM-L6-v2` encoder
- Token-aware context chunking
- Locked 1,543-dimensional primary feature representation
- Cosine-similarity threshold baseline
- Logistic Regression over the primary representation
- Style/length-only negative control
- Lightweight PyTorch MLP
- Three documented MLP seeds
- Validation-only tuning and threshold selection
- Locked final test evaluation
- Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices
- ROC and precision-recall curves
- Shuffled-context sanity test
- High-confidence false-positive and false-negative analysis
- Reproducible commands, configurations, manifests, and results
- Polished README with an honest results table and limitations

---

## 6. Optional Extensions

The following are optional and must not delay the core experiment:

- Question-context interaction vector blocks:
  - `|q - c_q|`
  - `q ⊙ c_q`
- Cluster-bootstrap confidence intervals
- More than three MLP seeds
- Calibration analysis
- Temperature scaling
- Brier score or expected calibration error
- Additional feature ablations
- Pairwise ranking accuracy
- Pair-consistency accuracy
- Near-duplicate semantic clustering
- Cross-dataset evaluation
- Minimal Streamlit or Gradio demo

Optional extensions must be:

- Clearly separated from primary results
- Added only after the core experiment is complete
- Documented as extensions or ablations
- Excluded if they threaten the project schedule

---

## 7. Dataset

### 7.1 Source

Use the official HaluEval QA dataset.

References:

- Repository: https://github.com/RUCAIBox/HaluEval
- Paper: https://aclanthology.org/2023.emnlp-main.397/

Expected original fields:

- `question`
- `knowledge`
- `right_answer`
- `hallucinated_answer`

Use an official revision-pinned dataset URL when available.

Record:

- Official repository
- Exact source revision
- Exact download URL
- Download date
- SHA-256 checksum
- Raw file size
- Raw record count
- Required schema
- License
- Paper citation

Raw downloaded data must not be silently modified.

### 7.2 Labels

The locked label convention is:

- `0 = Grounded`
- `1 = Hallucinated`

Hallucinated is the positive class for precision, recall, and F1.

### 7.3 Paired expansion

Every valid source record produces two examples:

1. `question + knowledge + right_answer -> Grounded`
2. `question + knowledge + hallucinated_answer -> Hallucinated`

Both examples must retain the same `source_id` and split assignment.

---

## 8. Data Validation and Stable IDs

Validate original records before splitting.

Detect and report:

- Non-object records
- Missing required fields
- Null fields
- Non-string fields
- Empty or whitespace-only fields
- Identical grounded and hallucinated answers after documented normalization
- Exact duplicate records
- Duplicate normalized questions
- Duplicate normalized contexts
- Duplicate normalized question-context pairs

Do not silently repair or exclude records.

Every exclusion must have a recorded reason.

If the source dataset has no stable ID, generate `source_id` as a deterministic SHA-256 hash of a canonical representation of the required original fields.

Requirements:

- Create IDs before splitting and expansion.
- Preserve original row index separately.
- Detect ID collisions.
- Do not derive IDs from absolute paths.
- Repeated runs over identical content must produce identical IDs.

Tracked duplicate artifacts should use hashes and identifiers rather than reproducing complete raw contexts or answers.

---

## 9. Shortcut-Artifact Audit

HaluEval correct and hallucinated answers originate from different generation processes. The model may learn answer style instead of grounding.

The core dataset audit must compare classes using:

- Answer character count
- Answer token count
- Answer sentence count
- Numeric-token presence
- Numeric-token count
- Answer-context lexical coverage
- Exact answer occurrence in context
- Answer-context semantic similarity when embeddings become available

The audit must include:

- Machine-readable results
- Human-readable summary
- Length-distribution visualization
- Lexical or semantic coverage visualization
- Conservative interpretation of observed differences

Standalone answer length and style features must not be included in the primary semantic feature representation.

They are reserved for the negative-control classifier.

---

## 10. Leakage-Safe Splitting

### 10.1 Required order

The mandatory order is:

1. Load original records.
2. Validate original records.
3. Assign stable source IDs.
4. Audit duplicates.
5. Create duplicate-context groups.
6. Split original grouped records.
7. Save a tracked split manifest.
8. Expand records within each split into paired labeled examples.

Never split expanded rows independently.

### 10.2 Split proportions

Use:

- 70% training
- 15% validation
- 15% test

Use the split seed from the canonical configuration.

Group-aware splitting may cause a small deviation from exact percentages. Record requested and achieved proportions.

### 10.3 Grouping rule

Records with identical normalized contexts must remain in the same split.

Do not group records merely because they share common grounded answers, such as common names, dates, “yes,” or “no.”

The split manifest must contain:

- `source_id`
- Original row index
- Duplicate-context group ID
- Assigned split

Tracked manifests must not contain full questions, contexts, or answers.

### 10.4 Required leakage assertions

Verify:

- No `source_id` crosses splits.
- No duplicate-context group crosses splits.
- Both paired examples from a source remain together.
- Every valid source appears exactly once in the original-record manifest.
- Every retained source generates exactly two examples.
- Every expanded split is exactly class-balanced.
- Labels retain their locked meanings.
- Example IDs are unique and deterministic.
- Repeated execution with the same inputs and seed reproduces the same manifest.

Expanded text datasets belong under ignored processed-data paths. Lightweight split manifests and metadata must remain trackable.

---

## 11. Frozen Embedding Pipeline

### 11.1 Encoder

Use:

- `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- L2-normalized embeddings
- Frozen encoder

Record package and model revision information where available.

CPU execution must be supported. GPU execution is optional.

### 11.2 Text preprocessing

Use minimal preprocessing:

- Convert validated fields to strings.
- Strip leading and trailing whitespace.
- Normalize repeated whitespace.
- Preserve case where meaningful.
- Preserve punctuation, numbers, and negation.
- Do not stem or lemmatize encoder input.
- Do not remove stopwords from encoder input.

Separate token normalization may be used for lexical coverage but must not alter encoder input.

### 11.3 Context chunking

Use tokenizer-aware chunks:

- Maximum chunk size: 192 model tokens
- Overlap: 32 model tokens

For each example:

- `q` = question embedding
- `a` = candidate-answer embedding
- `c1 ... cn` = context-chunk embeddings
- `c_a` = context chunk with maximum cosine similarity to the candidate answer

Also compute maximum question-context similarity as a scalar.

Full question-context interaction vectors are optional, not part of the primary architecture.

### 11.4 Caching

Cache embeddings and derived features.

Cache metadata must contain:

- Dataset checksum
- Split-manifest checksum
- Encoder name
- Encoder revision when available
- Embedding dimension
- Normalization setting
- Chunk size
- Chunk overlap
- Preprocessing version
- Feature schema version

Reject incompatible or stale caches clearly.

---

## 12. Primary Feature Representation

The primary vector contains four 384-dimensional interaction blocks:

1. `|a - c_a|`
2. `a ⊙ c_a`
3. `|q - a|`
4. `q ⊙ a`

Vector dimension:

`4 × 384 = 1,536`

Add seven scalar features:

1. Maximum answer-context-chunk cosine similarity
2. Mean of the top three answer-context-chunk similarities
3. Question-answer cosine similarity
4. Maximum question-context-chunk cosine similarity
5. Answer content-token coverage in context
6. Answer numeric-token coverage in context
7. Indicator that the answer contains a numeric token

Total primary feature dimension:

`1,536 + 7 = 1,543`

If fewer than three context chunks exist, average all available chunks for the top-three feature.

For answers containing no numeric tokens:

- Numeric coverage = `0`
- Numeric-presence indicator = `0`

Do not add standalone raw answer embeddings, answer length, or style features to the primary representation without approval.

Fit feature scaling using training data only.

Apply the fitted training scaler unchanged to validation, test, error-analysis, and demo inputs.

Logistic Regression and the MLP must use the same primary representation.

---

## 13. Models

### 13.1 Cosine baseline

Use maximum answer-context-chunk cosine similarity.

Expected direction:

- Higher similarity suggests grounded.
- Lower similarity suggests hallucinated.

Requirements:

- Select the threshold on validation data only.
- Optimize hallucinated-class validation F1.
- Use a correctly oriented continuous hallucination score for ROC-AUC.
- Do not adjust the threshold using test labels.

### 13.2 Logistic Regression

Train Logistic Regression using the 1,543-dimensional primary feature representation.

Use:

- L2 regularization
- No class weights unless an unexpected imbalance is documented
- Sufficient iterations for convergence
- Fixed random state where applicable

Validation-only regularization search:

- `C ∈ {0.01, 0.1, 1, 10}`

Select using hallucinated-class validation F1.

### 13.3 Style/length-only negative control

Train a separate lightweight Logistic Regression diagnostic using answer-only surface features.

Recommended features:

- Character count
- Token count
- Sentence count
- Numeric-token presence
- Numeric-token count
- Punctuation count or rate

The negative control must not receive:

- Question text or embeddings
- Context text or embeddings
- Answer-context coverage
- Answer-context similarity
- Primary semantic interaction features

Purpose:

- Measure how much label information can be recovered from answer-generation style alone.
- Diagnose shortcut learning.
- Inform limitations.

It is not one of the three primary models and must be labeled as a negative control.

### 13.4 Neural MLP

Locked architecture:

- Input: 1,543
- Linear: 1,543 -> 256
- ReLU
- Dropout: 0.20
- Linear: 256 -> 64
- ReLU
- Dropout: 0.10
- Linear: 64 -> 1

Training:

- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 128
- Maximum epochs: 30
- Early-stopping patience: 5
- Early-stopping criterion: validation loss
- Best checkpoint: lowest validation loss
- Classification threshold: validation-selected for hallucinated-class F1

Do not make the architecture more complex unless the locked model is technically defective and the change-control procedure is followed.

### 13.5 Seeds

Train the MLP using three documented seeds.

Report mean and standard deviation across the three runs.

For single-run visualizations and error analysis, select the representative run using validation performance only. Prefer the median validation-F1 run.

Never select the representative run using test performance.

---

## 14. Test-Set Protection

The test set is locked during development.

Use validation data for:

- Cosine threshold selection
- Logistic Regression regularization
- MLP early stopping
- Checkpoint selection
- MLP threshold selection
- Representative seed selection
- Optional architecture decisions

Do not use test data to decide:

- Which features to retain
- Which model configuration to use
- Whether an optional extension is beneficial
- Which seed to report
- Which threshold to use

If a real implementation defect is found after test evaluation, record:

- The defect
- Its effect
- The correction
- Why reevaluation was necessary

Do not silently replace results.

---

## 15. Evaluation Protocol

### 15.1 Primary metric

Primary metric:

- F1 for the hallucinated class

### 15.2 Required metrics

Report for the three primary approaches:

- Accuracy
- Precision for Hallucinated
- Recall for Hallucinated
- F1 for Hallucinated
- ROC-AUC
- Confusion matrix

Use continuous scores for ROC-AUC.

### 15.3 Required outputs

Produce:

- Model-comparison table
- Confusion matrices
- ROC curve
- Precision-recall curve
- MLP training/validation loss curve
- Machine-readable predictions
- Machine-readable metrics
- Recorded thresholds
- MLP seed summary

For the MLP, report mean and standard deviation across three seeds where appropriate.

### 15.4 Shuffled-context sanity test

After primary evaluation is locked, run a deterministic shuffled-context test.

Requirements:

- Shuffle contexts among test source records.
- Ensure a record does not retain its own context.
- Give both paired candidate answers from a source the same shuffled context.
- Do not retrain models.
- Recompute context-dependent features.
- Apply the existing training-fitted scaler, models, and validation-selected thresholds unchanged.
- Report performance changes for the cosine baseline, Logistic Regression, and representative MLP.
- Do not use the sanity-test results for model selection.

Interpretation:

- A context-grounding detector should normally deteriorate when contexts are unrelated.
- A small performance change may indicate reliance on answer style or other shortcuts.
- The outcome must be reported honestly; no predetermined drop is required.

---

## 16. Qualitative Error Analysis

Review high-confidence errors from the representative MLP.

Minimum:

- 10 high-confidence false positives
- 10 high-confidence false negatives

Definitions:

- False positive: grounded answer predicted hallucinated
- False negative: hallucinated answer predicted grounded

Record:

- `source_id`
- Question
- Relevant context excerpt
- Candidate answer
- True label
- Predicted label
- Hallucination score
- Distance from threshold
- Failure category
- Analyst note

Initial categories:

- Entity substitution
- Numerical hallucination
- Contradiction
- High lexical overlap but incorrect meaning
- Plausible but unsupported information
- Partial correctness
- Context ambiguity
- Evidence missed through context selection
- Dataset or label anomaly
- Other or unclear

Categories may be multi-label.

The analysis is descriptive and must not be presented as a population-level statistical estimate.

---

## 17. Reproducibility

The repository must support reproduction of:

1. Dataset acquisition
2. Validation and provenance
3. Artifact audit
4. Leakage-safe splitting
5. Paired expansion
6. Context chunking
7. Embedding generation
8. Feature generation
9. Baseline training
10. MLP training
11. Final evaluation
12. Shuffled-context testing
13. Error-analysis artifact generation
14. Figures and tables

Preserve:

- Canonical configuration
- Dataset checksum
- Source revision
- Split manifest
- Split-manifest checksum
- Seeds
- Cache metadata
- Feature schema
- Fitted scaler
- Model configurations
- Validation-selected thresholds
- Machine-readable results

Do not commit:

- Raw downloaded data
- Processed text datasets
- Embedding caches
- Virtual environments
- Secrets
- Temporary files
- Large model checkpoints unless explicitly required

Track lightweight provenance, manifests, metrics, figures, and error-analysis summaries when appropriate.

---

## 18. Project Phases

### Phase 0 — Research protocol and repository setup

Required outputs:

- Repository structure
- `PROJECT_PLAN.md`
- `AGENTS.md`
- Canonical configuration
- Dependency specification
- Packaging metadata
- `.gitignore`
- README skeleton
- Basic test infrastructure

Acceptance criteria:

- Package can be installed or imported as documented.
- Configuration is the single source of truth.
- Locked methodology is represented correctly.
- Test-set restrictions are documented.
- No later-phase implementation is introduced.

### Phase 1 — HaluEval acquisition and artifact audit

Required outputs:

- Official revision-pinned acquisition
- Provenance metadata
- Checksum
- Validation report
- Stable source-ID manifest
- Duplicate audit
- Text-level shortcut audit
- Audit figures and summary
- Synthetic network-free tests

Acceptance criteria:

- Official source and revision are recorded.
- All record exclusions are explained.
- Source IDs are deterministic.
- Duplicate counts are exact and internally consistent.
- Tracked audit artifacts do not reproduce the full raw dataset.
- Shortcut risks are documented.
- Tests pass.

### Phase 2 — Leakage-safe split and paired dataset

Required outputs:

- Duplicate-context grouping
- Original-record split manifest
- Split metadata and checksum
- Paired expanded train, validation, and test data
- Leakage-validation command
- Synthetic tests

Acceptance criteria:

- Split occurs before expansion.
- No source ID crosses splits.
- No duplicate-context group crosses splits.
- Both paired answers remain together.
- Expanded splits are balanced.
- Manifest generation is deterministic.
- Tracked manifests contain no full dataset text.
- Tests pass.

### Phase 3 — Context chunks, embeddings, and features

Required outputs:

- Frozen encoder wrapper
- Token-aware context chunking
- Embedding cache
- Answer-relevant chunk selection
- Seven scalar features
- 1,543-dimensional feature matrices
- Training-only scaler
- Cache validation
- Synthetic tests

Acceptance criteria:

- Encoder remains frozen.
- Feature dimensions equal 1,543.
- Scaling is fitted using training data only.
- Validation and test transformations reuse the training scaler.
- Cache metadata prevents incompatible reuse.
- No test-set model selection occurs.
- Tests pass.

### Phase 4 — Baselines and negative control

Required outputs:

- Cosine threshold baseline
- Logistic Regression baseline
- Style/length-only negative control
- Validation-only tuning
- Saved thresholds, models, and metrics
- Tests

Acceptance criteria:

- Primary Logistic Regression uses the shared 1,543 features.
- Negative control uses answer-only surface features.
- Validation data selects thresholds and regularization.
- ROC-AUC uses continuous scores.
- Test data does not guide decisions.
- Tests pass.

### Phase 5 — PyTorch MLP

Required outputs:

- Locked MLP architecture
- Reproducible training loop
- Early stopping
- Checkpoints
- Three seeded runs
- Training histories
- Validation-selected thresholds
- Tests

Acceptance criteria:

- Encoder remains frozen.
- Model input is 1,543.
- Validation loss controls early stopping.
- No test-set decisions occur.
- Three complete seeded runs exist.
- Tests pass.

### Phase 6 — Final evaluation and robustness

Required outputs:

- Locked test evaluation
- Required metrics and plots
- Comparison table
- MLP mean and standard deviation
- Representative validation-selected MLP run
- Shuffled-context sanity test
- Machine-readable predictions and results

Acceptance criteria:

- All primary models use the identical test examples.
- Metric direction and positive class are correct.
- Continuous scores are used for ROC-AUC.
- Shuffled-context evaluation does not retrain or retune models.
- Results are reported honestly.
- Tests pass.

### Phase 7 — Qualitative error analysis

Required outputs:

- High-confidence false-positive table
- High-confidence false-negative table
- At least 20 reviewed errors when available
- Failure categories
- Written findings

Acceptance criteria:

- At least 10 errors of each required type are reviewed when available.
- Evidence and rationale are preserved.
- Dataset anomalies are distinguished from model failures.
- Findings are clearly descriptive.

### Phase 8 — README, reproducibility, and optional demo

Required outputs:

- Complete README
- Results table
- Key figures
- Reproduction commands
- Limitations
- Citations
- Clean-environment verification

Optional output:

- Minimal Streamlit or Gradio demo

Acceptance criteria:

- A new user can reproduce the core experiment.
- README distinguishes primary models and negative control.
- Shortcut and shuffled-context results are discussed.
- Claims remain benchmark-specific.
- No fabricated values or unsupported claims appear.
- Demo is attempted only after the core project is complete.

---

## 19. Six-Day Scope

Original target schedule:

- Day 1: Phases 0–2
- Day 2: Phase 3
- Day 3: Phases 4–5
- Day 4: Phase 6
- Day 5: Phase 7 and README
- Day 6: Reproducibility, polish, and optional demo

When resuming an existing repository, do not restart this schedule. Determine completed phases and allocate remaining time only to incomplete phases.

If time becomes constrained, cut work in this order:

1. Demo
2. Calibration
3. Additional ablations
4. Confidence intervals
5. Extra seeds beyond three
6. Extended question-context vectors

Never cut:

- Leakage protection
- Three primary model comparison
- Negative control
- Shuffled-context sanity test
- Final evaluation
- Error analysis
- Reproducibility documentation

---

## 20. Definition of Done

The core project is complete when:

- Official HaluEval QA provenance is documented.
- Data validation and shortcut auditing are reproducible.
- Original records are split before paired expansion.
- Leakage assertions pass.
- Frozen chunked MiniLM embeddings are used.
- Primary features match the locked 1,543-dimensional schema.
- Cosine, Logistic Regression, and MLP are compared fairly.
- The style/length negative control is reported separately.
- Three MLP seeds are evaluated.
- Required metrics and plots are complete.
- The shuffled-context sanity test is complete.
- At least 20 high-confidence errors are reviewed when available.
- README contains methodology, results, limitations, citations, and reproduction steps.
- Conclusions remain faithful to the evidence.

A result in which the MLP does not win still satisfies the project objective.