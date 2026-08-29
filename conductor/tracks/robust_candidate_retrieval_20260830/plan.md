# Implementation Plan: Robust Multi-Provider Model Card Retrieval, Fallback Parsing & Baseline Seeding

## Phase 1: Dual-Mode Parsing & Domain Fallback in RetrieverAgent [checkpoint: 850aba7]
- [x] Task: Write tests for JSON regex parsing, markdown extraction, and domain fallbacks
    - [x] Add tests in `tests/search/test_retriever.py` for raw JSON text parsing, markdown extraction, and empty LLM fallback
- [x] Task: Implement dual-mode parser and domain fallback cards in `search/retriever.py`
    - [x] Implement `_parse_cards(raw_text: str)` extracting JSON blocks and markdown lists
    - [x] Implement `get_domain_fallback_cards(task_type)` for Recommender, Tabular, NLP, Vision modalities
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Baseline Code Seeding & Robust Initialization Pipeline [checkpoint: 7546785]
- [x] Task: Write tests for baseline script discovery and candidate injection
    - [x] Add tests in `tests/initialization/test_pipeline.py` verifying baseline injection
- [x] Task: Implement baseline script detection and injection in `initialization/pipeline.py`
    - [x] Check for `src/baseline/baseline.py` or workspace baseline script and add as Candidate 1
- [x] Task: Implement candidate status logging and merger fallback in `evaluator.py` and `merger.py`
    - [x] Print `[Candidate {i}/{M}] {name} -> Score: {score}` with live unbuffered flush
    - [x] Ensure `ModelMergerAgent` always preserves the highest scoring individual candidate if merging fails
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Full System Integration & Regression Verification
- [x] Task: Run full test suite and verify 100% pass rate
    - [x] Execute `uv run pytest --tb=short -q` across all 310+ tests (346 passed)
- [~] Task: Verify end-to-end initialization on KuaiRand-Pure.md
    - [ ] Run dry-run and mock test verifying candidate evaluation and baseline score generation
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
