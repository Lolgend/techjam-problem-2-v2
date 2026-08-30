"""Unit tests for the unified competition run-log contract and central logger.

Covers the ``IterationLogEntry`` schema (all 13 spec fields, defaults, and
the backward-compatible ``plan``/``errors`` aliases) and the thread-safe
``CentralIterationLogger`` JSONL writer.
"""

import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from problem_2_v2.contracts.iteration import CentralIterationLogger, IterationLogEntry


def _entry(**overrides: object) -> IterationLogEntry:
    base = {
        "iteration_id": "cand_1",
        "stage": "INITIALIZATION",
        "hypothesis": "Try a gradient boosted model for better ranking.",
        "code_diff": "--- old_code\n+++ new_code\n@@ -1 +1 @@\n",
        "metrics": {"primary": 0.7381},
        "validation_score": 0.7381,
        "delta_from_baseline": 0.1365,
        "error_recovery_events": ["syntax error repaired by the debugger"],
        "success": True,
        "target_component": "MODEL_ARCHITECTURE",
        "branch_index": 0,
        "duration_seconds": 12.3,
    }
    base.update(overrides)
    return IterationLogEntry(**base)


class TestIterationLogEntrySchema:
    """Test the 13-field competition run-log schema."""

    def test_requires_hypothesis_and_error_recovery_events(self) -> None:
        with pytest.raises(ValidationError):
            IterationLogEntry(iteration_id="x", stage="INITIALIZATION", success=True)

    def test_optional_fields_have_defaults(self) -> None:
        entry = IterationLogEntry(
            iteration_id="cand_1",
            stage="INITIALIZATION",
            hypothesis="rationale",
            success=True,
        )
        assert entry.metrics == {}
        assert entry.error_recovery_events == []
        assert entry.code_diff == ""
        assert entry.validation_score is None
        assert entry.delta_from_baseline is None
        assert entry.target_component is None
        assert entry.branch_index is None
        assert entry.duration_seconds is None

    def test_all_spec_fields_round_trip(self) -> None:
        entry = _entry()
        assert entry.iteration_id == "cand_1"
        assert entry.stage == "INITIALIZATION"
        assert entry.hypothesis.startswith("Try a gradient boosted")
        assert "+++ new_code" in entry.code_diff
        assert entry.metrics == {"primary": 0.7381}
        assert entry.validation_score == pytest.approx(0.7381)
        assert entry.delta_from_baseline == pytest.approx(0.1365)
        assert entry.error_recovery_events == ["syntax error repaired by the debugger"]
        assert entry.success is True
        assert entry.target_component == "MODEL_ARCHITECTURE"
        assert entry.branch_index == 0
        assert entry.timestamp is not None
        assert entry.duration_seconds == pytest.approx(12.3)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            _entry(unexpected_field="nope")

    def test_json_serializes_canonical_field_names(self) -> None:
        entry = _entry()
        data = json.loads(entry.model_dump_json())
        assert data["hypothesis"] == entry.hypothesis
        assert data["error_recovery_events"] == entry.error_recovery_events
        assert "plan" not in data
        assert "errors" not in data

    def test_round_trip_via_model_validate_json(self) -> None:
        entry = _entry()
        restored = IterationLogEntry.model_validate_json(entry.model_dump_json())
        assert restored == entry

    def test_backward_compatible_plan_and_errors_aliases(self) -> None:
        legacy = IterationLogEntry(
            iteration_id="merge_1",
            stage="INITIALIZATION",
            plan="merge the two models",
            code_diff="",
            errors=["merge rejected: score regression"],
            success=False,
        )
        assert legacy.hypothesis == "merge the two models"
        assert legacy.plan == "merge the two models"
        assert legacy.error_recovery_events == ["merge rejected: score regression"]
        assert legacy.errors == ["merge rejected: score regression"]
        data = json.loads(legacy.model_dump_json())
        assert data["hypothesis"] == "merge the two models"
        assert data["error_recovery_events"] == ["merge rejected: score regression"]


class TestCentralIterationLogger:
    """Test the thread-safe JSONL iteration logger."""

    def test_for_run_resolves_logs_path(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger.for_run(str(tmp_path), "run_1")
        assert logger.logs_path == (tmp_path / "run_1" / "iteration_logs.jsonl").resolve()

    def test_append_creates_parent_directories(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger(tmp_path / "nested" / "deep" / "iteration_logs.jsonl")
        logger.append(_entry())
        assert logger.logs_path.is_file()

    def test_append_writes_one_json_line_per_entry(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger.for_run(str(tmp_path), "r")
        logger.append(_entry(iteration_id="a"))
        logger.append(_entry(iteration_id="b", stage="REFINEMENT"))
        lines = logger.logs_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["iteration_id"] == "a"
        assert json.loads(lines[1])["stage"] == "REFINEMENT"

    def test_read_all_returns_parsed_entries(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger.for_run(str(tmp_path), "r")
        logger.append(_entry(iteration_id="ens_r0", stage="ENSEMBLING"))
        entries = logger.read_all()
        assert len(entries) == 1
        assert isinstance(entries[0], IterationLogEntry)
        assert entries[0].stage == "ENSEMBLING"

    def test_read_all_empty_when_file_missing(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger.for_run(str(tmp_path), "missing")
        assert logger.read_all() == []

    def test_concurrent_appends_are_thread_safe(self, tmp_path: Path) -> None:
        logger = CentralIterationLogger.for_run(str(tmp_path), "r")
        threads = 8
        per_thread = 25

        def worker(tid: int) -> None:
            for i in range(per_thread):
                logger.append(
                    _entry(iteration_id=f"t{tid}_{i}", success=True)
                )

        pool = [threading.Thread(target=worker, args=(tid,)) for tid in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

        entries = logger.read_all()
        assert len(entries) == threads * per_thread
        assert len({e.iteration_id for e in entries}) == threads * per_thread
