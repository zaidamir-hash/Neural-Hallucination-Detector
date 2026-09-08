# AGENTS.md

## 1. Repository Status

This is an existing research repository.

Do not assume it is empty or newly created.

Before modifying anything:

1. Read `PROJECT_PLAN.md` completely.
2. Read this file completely.
3. Inspect the current repository.
4. Inspect Git status and recent history.
5. Identify existing source files, tests, configurations, manifests, and results.
6. Determine the completion status of every project phase.
7. Report the current state before making changes.
8. Wait for confirmation before implementing the next phase.

Preserve valid existing work. Do not recreate completed phases without evidence that they are defective.

## 2. Authority

`PROJECT_PLAN.md` is the methodological source of truth.

`AGENTS.md` defines implementation conduct.

The current user prompt defines the authorized phase or inspection task.

If the current code, configuration, user prompt, `PROJECT_PLAN.md`, and this file conflict, stop and report:

1. The conflicting requirements
2. Existing repository evidence
3. The smallest proposed resolution
4. Methodological consequences

Do not silently resolve methodology conflicts.

## 3. Role of the Implementation Agent

The implementation agent may:

- Inspect the repository
- Implement the explicitly authorized phase
- Add or modify tests within that phase
- Run relevant commands and tests
- Update documentation for completed factual work
- Fix defects within the authorized phase
- Report results and blockers

The implementation agent must not:

- Begin unrequested phases
- Rewrite completed valid work
- Change the research protocol silently
- Fabricate metrics, results, files, or command outcomes
- Use test results for development decisions
- Implement optional extensions without authorization

## 4. Phase Discipline

- Work on one explicitly authorized phase at a time.
- Read the phase and acceptance criteria before starting.
- Inspect dependencies from earlier phases.
- Reuse valid existing utilities.
- Avoid unrelated refactors.
- Stop after the requested phase.
- Report completion before committing when review is requested.
- Do not automatically continue to the next phase.

File presence alone does not establish phase completion. Verify behavior, tests, and artifacts.

## 5. Locked Methodology

Do not silently change:

- Dataset: official HaluEval QA
- Label convention: `0 = Grounded`, `1 = Hallucinated`
- Positive class: Hallucinated
- Split proportions: 70/15/15
- Split seed from the canonical configuration
- Split original records before expansion
- Duplicate normalized contexts remain in one split
- Both paired examples remain in one split
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Encoder remains frozen
- Embedding dimension: 384
- L2 embedding normalization
- Context chunk size: 192 model tokens
- Context overlap: 32 model tokens
- Four primary interaction blocks
- Seven primary scalar features
- Total primary dimension: 1,543
- Same primary features for Logistic Regression and MLP
- MLP architecture: `1543 -> 256 -> 64 -> 1`
- ReLU activations
- Dropout: 0.20 and 0.10
- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 128
- Maximum epochs: 30
- Early-stopping patience: 5
- Early-stopping criterion: validation loss
- Three core MLP seeds
- Primary metric: hallucinated-class F1
- Validation-only tuning
- Locked test-set policy
- Style/length-only negative control
- Shuffled-context sanity test
- Qualitative high-confidence error analysis

## 6. Primary Feature Contract

The primary feature vector is:

- `|a - c_a|`
- `a ⊙ c_a`
- `|q - a|`
- `q ⊙ a`

Each block has 384 dimensions.

Required scalar features:

1. Maximum answer-context-chunk cosine similarity
2. Mean top-three answer-context-chunk similarity
3. Question-answer cosine similarity
4. Maximum question-context-chunk similarity
5. Answer content-token coverage in context
6. Answer numeric-token coverage in context
7. Numeric-token presence indicator

Total dimension: 1,543.

Do not add length or style features to this primary representation.

Question-context vector interactions are optional and require explicit authorization.

## 7. Leakage Prevention

The required processing order is:

1. Validate original records.
2. Assign stable source IDs.
3. Audit duplicates.
4. Group duplicate normalized contexts.
5. Split original records.
6. Save the split manifest.
7. Expand records within their assigned splits.

Never split expanded rows independently.

Required assertions:

- No source ID crosses splits.
- No duplicate-context group crosses splits.
- Both variants of each source remain together.
- Every valid source appears exactly once in the manifest.
- Every source produces exactly two expanded rows.
- Each expanded split is balanced.
- Example IDs are unique.
- Manifest generation is deterministic.

## 8. Test-Set Protection

Never use test data to:

- Select features
- Fit preprocessing
- Fit the scaler
- Select cosine thresholds
- Select Logistic Regression regularization
- Tune the MLP
- Control early stopping
- Select checkpoints
- Select classification thresholds
- Select the representative seed
- Decide whether an extension should be retained

Use:

- Training data for fitting
- Validation data for selection and thresholds
- Test data only for locked final evaluation and prespecified sanity analysis

Do not repeatedly inspect test results during development.

## 9. Core Diagnostics

### Style/length negative control

The negative control must use answer-only surface features.

It must not receive question or context information.

Report it separately from the three primary models.

### Shuffled-context test

Run only after primary evaluation is locked.

Do not retrain or retune models.

Preserve answer pairs and ensure no source retains its own context.

A performance drop is expected but not required. Report the observed result honestly.

## 10. Optional Work

The following require explicit authorization:

- Extended question-context vector interactions
- Bootstrap confidence intervals
- More than three MLP seeds
- Calibration
- Extra ablations
- Pairwise metrics
- Near-duplicate semantic clustering
- Demo
- Deployment

Do not implement optional work simply because time appears available.

## 11. Implementation Standards

- Use Python.
- Use PyTorch for the MLP.
- Use sentence-transformers for embeddings.
- Use scikit-learn for baselines and evaluation utilities.
- Use pandas and NumPy for data processing.
- Use matplotlib or another lightweight plotting tool.
- Keep reusable logic under `src/hallucination_detector`.
- Keep scripts thin; scripts should call package functions.
- Do not place the reproducible pipeline only in notebooks.
- Centralize configuration.
- Avoid duplicated constants.
- Prefer simple functions and modules.
- Add useful type hints and concise docstrings.
- Validate external data at boundaries.
- Raise informative errors.
- Avoid unnecessary dependencies and frameworks.

## 12. Existing Work Protection

Before editing an existing file:

1. Read it.
2. Understand its callers and tests.
3. Preserve valid behavior.
4. Avoid formatting-only rewrites.
5. Avoid unrelated refactors.
6. Explain necessary structural changes.

Never overwrite `PROJECT_PLAN.md` with a summary.

Never weaken `AGENTS.md`.

Never delete valid artifacts merely because they were created by another agent.

If an existing implementation satisfies an acceptance criterion, reuse it.

## 13. Testing

- Add or update tests for every implemented phase.
- Use small synthetic fixtures for routine tests.
- Routine tests must not require downloading HaluEval or MiniLM.
- Mock network access where appropriate.
- Do not weaken valid tests to make changes pass.
- Report skipped tests and warnings.
- Run the complete relevant test suite before declaring completion.

Prioritize tests for:

- Schema validation
- Stable IDs
- Duplicate grouping
- Split determinism
- Leakage assertions
- Label correctness
- Context chunk boundaries
- Feature dimension
- Train-only scaling
- Cache compatibility
- Metric direction
- Threshold selection
- MLP shape and determinism
- Shuffled-context integrity

## 14. Git and Artifact Safety

Before committing:

- Inspect `git status`.
- Review every staged file.
- Confirm raw data is ignored.
- Confirm secrets and environments are ignored.
- Confirm caches and large checkpoints are ignored.
- Keep lightweight provenance, manifests, metrics, and figures trackable.
- Do not commit generated junk or machine-specific absolute paths.

Do not rewrite history, force-push, delete branches, or remove user work without explicit authorization.

## 15. Change-Control Procedure

If a locked decision appears defective:

1. Stop before changing it.
2. Identify the exact requirement.
3. Provide evidence.
4. Propose the smallest correction.
5. Explain its methodological impact.
6. Explain which artifacts would need regeneration.
7. Wait for approval.

Minor implementation choices that do not alter methodology may be made and reported normally.

## 16. Required Status Report

For repository inspection, report:

1. Repository root and current branch
2. Git status and recent relevant commits
3. Existing files and directory structure
4. Canonical configuration values
5. Status of every phase:
   - Complete
   - Partial
   - Not started
   - Unclear
6. Evidence supporting each status
7. Existing tests and results
8. Test execution outcome, if safely run
9. Methodology inconsistencies or defects
10. First incomplete phase
11. Recommended next action

Do not modify files during an inspection-only task.

## 17. Required Phase Completion Report

After implementing an authorized phase, report:

1. Files created, modified, or deleted
2. Existing files reused
3. Important implementation decisions
4. Commands executed
5. Tests and outcomes
6. Generated artifacts and results
7. Acceptance criteria satisfied
8. Acceptance criteria not satisfied
9. Assumptions
10. Warnings or unresolved issues
11. Git status
12. Recommended next phase

Stop after reporting.