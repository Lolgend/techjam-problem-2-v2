"""Task ingestion: markdown problem description to validated `TaskSpecification`.

Deterministic parsing is used to extract structured metadata (metric direction,
baseline score, target variable, dataset files, etc.) while preserving the
full raw markdown description in ``TaskSpecification.raw_description`` for direct
injection into downstream agent prompts.
"""

from __future__ import annotations

import logfire

from problem_2_v2.contracts.task import TaskSpecification


class TaskExtractor:
    """Ingests markdown problem descriptions into validated task specs."""

    def __init__(self, model: str | None = None, use_llm: bool = False) -> None:
        """Create a task extractor.

        Args:
            model: Optional model identifier (retained for signature compatibility).
            use_llm: Retained for signature compatibility; extraction is deterministic.
        """
        self.model = model
        self.use_llm = use_llm

    def extract(self, md_text: str, dataset_dir: str) -> TaskSpecification:
        """Extract a validated task specification from markdown.

        Args:
            md_text: The raw markdown problem description.
            dataset_dir: Absolute path to the dataset directory.

        Returns:
            A validated ``TaskSpecification`` containing parsed metadata and
            the full ``raw_description``.
        """
        with logfire.span("task_extractor.extract"):
            return TaskSpecification.from_markdown(md_text, dataset_dir=dataset_dir)
