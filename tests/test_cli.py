"""Unit tests for the ``problem-2-v2`` command-line interface.

Covers the ``run`` subcommand flags, ``--dry-run`` validation, the
``version`` subcommand, invalid argument handling, and API-key env wiring.
"""

import os
from pathlib import Path

import pytest

from problem_2_v2 import main
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.execution.finalizer import FinalArtifact
from problem_2_v2.orchestrator import MLEStarResult

_MD = (
    "**Task Name:** Demo\n"
    "**Task Type:** TABULAR_CLASSIFICATION\n"
    "**Metric Name:** AUROC\n"
    "**Metric Direction:** MAXIMIZE\n"
    "**Target Variable:** label\n"
    "**Baseline Score:** 0.50\n"
    "**Dataset Files:** train.csv\n"
    "**Description:** classify.\n"
)


def _write_task(tmp_path: Path) -> tuple[Path, Path]:
    task_file = tmp_path / "problem.md"
    task_file.write_text(_MD, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    return task_file, data_dir


class TestCLI:
    """Test the CLI entry point and argument handling."""

    def test_version_command(self, capsys) -> None:
        code = main(["version"])
        out = capsys.readouterr().out
        assert code == 0
        assert "problem-2-v2" in out
        assert "Python" in out

    def test_run_dry_run(self, tmp_path: Path, capsys) -> None:
        task_file, data_dir = _write_task(tmp_path)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--search-provider",
                "mock",
                "--dry-run",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Demo" in out

    def test_run_requires_task(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with pytest.raises(SystemExit):
            main(["run", "--data", str(data_dir)])

    def test_run_rejects_invalid_search_provider(self, tmp_path: Path, capsys) -> None:
        task_file, data_dir = _write_task(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "run",
                    "--task",
                    str(task_file),
                    "--data",
                    str(data_dir),
                    "--search-provider",
                    "bogus",
                ]
            )

    def test_run_rejects_seed_branch_mismatch(self, tmp_path: Path, capsys) -> None:
        task_file, data_dir = _write_task(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "run",
                    "--task",
                    str(task_file),
                    "--data",
                    str(data_dir),
                    "--search-provider",
                    "mock",
                    "--branches",
                    "4",
                    "--seeds",
                    "1,2",
                ]
            )

    def test_run_failed_result_exits_cleanly(self, tmp_path: Path, capsys, monkeypatch) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def failed_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=None,
                baseline_score=0.5,
                final_score=None,
                score_delta=None,
                duration_seconds=1.0,
                success=False,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", failed_run)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--search-provider",
                "mock",
            ]
        )
        captured = capsys.readouterr()
        assert code == 1
        assert "Final: n/a" in captured.out
        assert "Delta: n/a" in captured.out
        assert "TypeError" not in captured.err

    def test_run_dry_run_rejects_missing_data(self, tmp_path: Path, capsys) -> None:
        task_file, _ = _write_task(tmp_path)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(tmp_path / "no_data"),
                "--search-provider",
                "mock",
                "--dry-run",
            ]
        )
        assert code != 0
        assert "Dataset directory not found" in capsys.readouterr().err

    def test_run_launches_pipeline(self, tmp_path: Path, capsys, monkeypatch) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))
        captured: dict[str, object] = {}

        def fake_run(self, task, data, run_id=None):
            captured["task"] = task
            captured["data"] = data
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=FinalArtifact(
                    code="print(1)",
                    output_dir=str(tmp_path / "out"),
                    model_paths=[],
                    metrics={"auroc": 0.9},
                    submission_path=None,
                    validation_score=0.9,
                    success=True,
                ),
                baseline_score=0.5,
                final_score=0.9,
                score_delta=0.4,
                duration_seconds=1.5,
                success=True,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", fake_run)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--search-provider",
                "mock",
                "--output",
                str(tmp_path / "final_out"),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert captured["task"] == str(task_file)
        assert "Delta" in out

    def test_run_sets_api_key_env_var(self, tmp_path: Path, capsys, monkeypatch) -> None:
        import os

        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def fake_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=None,
                baseline_score=0.5,
                final_score=0.8,
                score_delta=0.3,
                duration_seconds=1.0,
                success=True,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", fake_run)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--model",
                "anthropic:claude-3-7-sonnet",
                "--api-key",
                "sk-ant-test-key",
                "--search-provider",
                "tavily",
                "--search-api-key",
                "tvly-test-key",
            ]
        )
        assert code == 0
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-key"
        assert os.environ.get("TAVILY_API_KEY") == "tvly-test-key"
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("TAVILY_API_KEY", None)

    @pytest.mark.parametrize(
        ("model", "expected_key"),
        [
            ("deepseek:deepseek-chat", "DEEPSEEK_API_KEY"),
            ("google:gemini-2.0-flash", "GEMINI_API_KEY"),
            ("openrouter:openai/gpt-4o", "OPENROUTER_API_KEY"),
            ("groq:llama-3", "GROQ_API_KEY"),
            ("mistral:mistral-large", "MISTRAL_API_KEY"),
            ("openai:gpt-4o", "OPENAI_API_KEY"),
        ],
    )
    def test_api_key_mapping_sets_correct_env(
        self, tmp_path: Path, capsys, monkeypatch, model: str, expected_key: str
    ) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def fake_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=None,
                baseline_score=0.5,
                final_score=0.8,
                score_delta=0.3,
                duration_seconds=1.0,
                success=True,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", fake_run)
        for key in (
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--model",
                model,
                "--api-key",
                "sk-test",
                "--search-provider",
                "mock",
            ]
        )
        assert code == 0
        assert os.environ.get(expected_key) == "sk-test"

    def test_base_url_sets_env(self, tmp_path: Path, capsys, monkeypatch) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def fake_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=None,
                baseline_score=0.5,
                final_score=0.8,
                score_delta=0.3,
                duration_seconds=1.0,
                success=True,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", fake_run)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--base-url",
                "https://api.deepseek.com",
                "--search-provider",
                "mock",
            ]
        )
        assert code == 0
        assert os.environ.get("OPENAI_BASE_URL") == "https://api.deepseek.com"
        assert os.environ.get("DEEPSEEK_BASE_URL") == "https://api.deepseek.com"

    def test_google_search_api_key_sets_env(self, tmp_path: Path, capsys, monkeypatch) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def fake_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=None,
                baseline_score=0.5,
                final_score=0.8,
                score_delta=0.3,
                duration_seconds=1.0,
                success=True,
            )

        monkeypatch.setattr("problem_2_v2.orchestrator.MLEStarPipeline.run", fake_run)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CSE_ID", "test-cx")
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--search-provider",
                "google",
                "--search-api-key",
                "g-key",
            ]
        )
        assert code == 0
        assert os.environ.get("GOOGLE_API_KEY") == "g-key"
