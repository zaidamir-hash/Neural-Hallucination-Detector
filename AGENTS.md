# AGENTS.md

## Project Authority

Before modifying this repository, read `PROJECT_PLAN.md` completely.

`PROJECT_PLAN.md` is the methodological source of truth. This file defines the operating rules for coding agents.

If this file, a user request, existing code, and `PROJECT_PLAN.md` appear inconsistent, stop and report the conflict. Do not silently choose an interpretation or change the research protocol.

## Project Goal

Build a focused and reproducible research experiment that compares:

1. An answer-context cosine-similarity threshold
2. Logistic Regression using engineered embedding features
3. A lightweight PyTorch MLP using the same primary features

The project is limited to the HaluEval QA benchmark.

The objective is experimental validity, reproducibility, and honest comparison—not obtaining a preferred result.

## Phase Discipline

- Work only on the phase explicitly named in the current prompt.
- Read that phase and its acceptance criteria in `PROJECT_PLAN.md` before beginning.
- Do not implement future phases early.
- Stop after completing the requested phase.
- Do not automatically continue to the next phase.
- Keep the implementation achievable within six days.
- Prefer the smallest design that satisfies the current acceptance criteria.
- Optional extensions are out of scope unless explicitly requested.
- Do not prioritize the demo over the research experiment.

## Locked Research Decisions

The following decisions are locked and must not be changed silently:

- Dataset: official HaluEval QA subset
- Label convention: `0 = Grounded`, `1 = Hallucinated`
- Positive evaluation class: Hallucinated
- Split proportions: 70% train, 15% validation, 15% test
- Split original records before paired expansion
- Keep both answers from every source record in the same split
- Use a fixed and documented split seed
- Sentence encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Sentence encoder remains frozen
- Embedding dimension: 384
- Context chunk size: 192 model tokens
- Context chunk overlap: 32 model tokens
- Primary feature dimension: 1,543
- Primary vector features:
  - `|a - ca|`
  - `a element-wise-product ca`
  - `|q - a|`
  - `q element-wise-product a`
- Primary scalar features:
  - Maximum answer-context-chunk cosine similarity
  - Mean of the top three answer-context-chunk similarities
  - Question-answer cosine similarity
  - Maximum question-context-chunk cosine similarity
  - Answer content-token coverage in the context
  - Answer numeric-token coverage in the context
  - Indicator that the answer contains a numeric token
- MLP architecture: `1543 -> 256 -> 64 -> 1`
- Hidden activation: ReLU
- Dropout values: 0.20 and 0.10
- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW
- Initial learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 128
- Maximum epochs: 30
- Early-stopping patience: 5
- Early-stopping criterion: validation loss
- Number of core MLP seeds: 3
- Primary metric: hallucinated-class F1
- Test data must not guide development decisions

Question-context interaction vector blocks are optional and are not part of the locked primary architecture.

Cluster-bootstrap confidence intervals, calibration, additional ablations, extra MLP seeds, shuffled-context testing, and the demo are optional extensions.

## Leakage Prevention

The required dataset order is:

1. Load original HaluEval records.
2. Validate original records.
3. Assign stable source identifiers.
4. Audit duplicates.
5. Split original records into train, validation, and test groups.
6. Expand each split into grounded and hallucinated rows.

Never split the expanded classification rows independently.

Automated tests must verify:

- No source identifier crosses splits.
- Both expanded examples from a source record remain together.
- Every retained source record creates exactly two rows.
- No known exact-duplicate group crosses splits.
- Every split remains balanced.
- Reusing the same seed reproduces the same split manifest.

## Test-Set Protection

Never use test data to:

- Select features
- Fit a scaler or preprocessing transformation
- Select the cosine threshold
- Select Logistic Regression regularization
- Select MLP hyperparameters
- Control early stopping
- Select a checkpoint
- Select a classification threshold
- Select the representative MLP seed
- Decide whether an optional extension should be retained

Use training data to fit model parameters and fitted transformations.

Use validation data for model selection, threshold selection, and early stopping.

Use test data only for locked final evaluation.

If a genuine implementation defect is discovered after final test evaluation, document:

1. The defect
2. Its effect
3. The correction
4. Why reevaluation was necessary

Do not silently replace results.

## Shortcut Awareness

Correct and hallucinated HaluEval answers may differ systematically in length and style.

The core dataset audit must examine possible shortcut features, including:

- Answer token count
- Answer character count
- Sentence count
- Numeric-token frequency
- Answer-context lexical coverage
- Answer-context cosine similarity
- Exact answer occurrence in the context

Standalone answer length and stylistic features must not be added to the primary feature vector.

Do not interpret benchmark performance as proof of general real-world hallucination detection.

## Implementation Standards

- Use Python.
- Use PyTorch for the MLP.
- Use sentence-transformers for frozen embeddings.
- Use scikit-learn for baselines, scaling, and evaluation utilities.
- Use pandas and NumPy for data processing.
- Use matplotlib or compatible lightweight plotting tools for figures.
- Keep reusable logic inside `src/hallucination_detector`.
- Do not place the reproducible pipeline exclusively in notebooks.
- Use notebooks only for focused exploration or presentation.
- Centralize experiment settings in a configuration file.
- Avoid duplicated configuration constants.
- Use type hints where they improve clarity.
- Add concise docstrings to public functions and classes.
- Validate data at external boundaries.
- Raise informative errors for malformed records and incompatible caches.
- Prefer simple functions and modules over unnecessary abstractions.
- Avoid adding dependencies without a clear project requirement.

## Prohibited Additions

Do not add:

- Paid APIs
- External LLM API calls
- Cloud-only services
- Databases
- Distributed training infrastructure
- Large-model fine-tuning
- Unrequested web services
- Production deployment infrastructure
- Complex configuration frameworks
- Large hyperparameter-search frameworks
- Unrequested optional extensions
- Invented metrics or results

## Repository Hygiene

Do not commit:

- Downloaded datasets unless their license and project policy explicitly permit it
- Virtual environments
- Python caches
- Embedding caches
- Model checkpoints
- Temporary files
- Secrets or API keys
- Local IDE state
- Generated artifacts that are large and reproducible

Keep generated results separate from source code.

Preserve:

- Dataset provenance
- Dataset checksum
- Split manifest
- Random seeds
- Experiment configuration
- Cache metadata
- Selected validation thresholds
- Model settings
- Machine-readable evaluation results

## Testing Rules

- Add or update tests for every implemented phase.
- Use small synthetic fixtures for routine unit tests.
- Routine unit tests must not require downloading HaluEval.
- Routine unit tests must not require downloading the sentence-transformer model.
- Mock or inject heavyweight external components where appropriate.
- Run relevant tests before declaring a phase complete.
- Report skipped tests and explain why they were skipped.
- Do not weaken or delete valid tests merely to make a task pass.

Prioritize tests for:

- Label consistency
- Stable source identifiers
- Split reproducibility
- Leakage prevention
- Paired expansion
- Feature dimensions
- Context chunking
- Train-only fitting
- Cache compatibility
- Metric direction
- Deterministic behavior

## Existing Work

Before modifying existing files:

1. Inspect their current contents.
2. Preserve valid work from earlier phases.
3. Avoid unrelated rewrites.
4. Explain any necessary structural change.
5. Do not overwrite `PROJECT_PLAN.md` with a summary.
6. Do not weaken `AGENTS.md`.

Do not edit completed experimental results unless the current task explicitly requires it.

## Change-Control Procedure

If a locked requirement appears technically impossible, internally inconsistent, or clearly defective, stop before changing it.

Report:

1. The affected requirement
2. Evidence demonstrating the problem
3. The smallest proposed correction
4. Its effect on methodology and reproducibility
5. Whether existing artifacts would need regeneration

Wait for explicit approval before making the change.

Minor implementation details that do not alter the methodology may be decided normally and documented in the completion report.

## Required Phase Completion Report

At the end of every phase, stop and report:

1. Files created or changed
2. Important implementation decisions
3. Commands executed
4. Tests executed and their outcomes
5. Acceptance criteria satisfied
6. Acceptance criteria not satisfied
7. Assumptions made
8. Warnings or unresolved issues
9. Recommended next phase

Do not begin the recommended next phase until explicitly requested.