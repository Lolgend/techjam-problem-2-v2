"""Task ingestion agent: markdown problem description to `TaskSpecification`.

The extractor first attempts an LLM-backed structured extraction via a
Pydantic AI agent. When the LLM is unavailable, disabled, or returns an
invalid specification, it falls back to the deterministic heuristic
markdown parser in ``contracts.task``.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.task import TaskSpecification

_EXTRACT_INSTRUCTIONS = (
    "You parse machine learning competition problem descriptions written in "
    "Markdown into a structured task specification.\n"
    "- Map the described task to one of the allowed TaskType values.\n"
    "- Determine the evaluation metric name and whether higher (MAXIMIZE) or "
    "lower (MINIMIZE) scores are better.\n"
    "- Identify the target (label) variable and list every dataset file "
    "mentioned.\n"
    "- Extract the official baseline score and any constraints verbatim.\n"
    "- The dataset_dir field is provided by the caller; use it as-is."
)


class TaskExtractor:
    """Ingests markdown problem descriptions into validated task specs.

    Attributes:
        agent: The Pydantic AI agent bound to structured ``TaskSpecification``
            output.
        use_llm: Whether to attempt LLM extraction before the heuristic
            fallback.
    """

    def __init__(self, model: str = "openai:gpt-4o", use_llm: bool = True) -> None:
        """Create a task extractor.

        Args:
            model: Pydantic AI model string (e.g. ``google:gemini-2.0-flash``).
            use_llm: If False, extraction always uses the heuristic parser
                (offline mode).
        """
        self.use_llm = use_llm
        self.agent = Agent(
            model,
            name="task_extractor",
            output_type=TaskSpecification,
            instructions=_EXTRACT_INSTRUCTIONS,
            defer_model_check=True,
        )

    def extract(self, md_text: str, dataset_dir: str) -> TaskSpecification:
        """Extract a validated task specification from markdown.

        Args:
            md_text: The raw markdown problem description.
            dataset_dir: Absolute path to the dataset directory.

        Returns:
            A validated ``TaskSpecification``. The dataset directory is
            always taken from the ``dataset_dir`` argument, never from the
            model output.
        """
        if self.use_llm:
            try:
                with logfire.span("task_extractor.llm_extract"):
                    result = self.agent.run_sync(
                        f"# Problem description\n{md_text}\n# Dataset directory\n{dataset_dir}"
                    )
                spec = result.output
                if spec.dataset_dir != dataset_dir:
                    spec.dataset_dir = dataset_dir
                return spec
            except Exception:
                logfire.warn("task_extractor.llm_extract.failed; falling back to heuristic parser")
        with logfire.span("task_extractor.heuristic_extract"):
            return TaskSpecification.from_markdown(md_text, dataset_dir=dataset_dir)
