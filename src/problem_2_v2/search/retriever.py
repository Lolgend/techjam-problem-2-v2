"""Retriever agent ($A_retriever$): search-guided candidate model retrieval.

Formulates a targeted query from the task specification, fetches web
results through a pluggable ``SearchProvider``, and prompts the LLM
(Figure 9 prompt) to distill the search context into structured model
cards.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.search import ModelCard, RetrievedCandidates
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.search.providers import SearchProvider

_RETRIEVER_INSTRUCTIONS = (
    "You are retrieving state-of-the-art models for a machine learning "
    "competition.\n"
    "# Your task\n"
    "- List recent effective models and their example codes to win the "
    "competition described below.\n"
    "# Requirement\n"
    "- The example code should be concise and simple.\n"
    "- You must provide an example code, i.e., do not just mention "
    "GitHubs or papers.\n"
    "- Ground your suggestions in the web search results provided.\n"
)


class RetrieverAgent:
    """Retrieves candidate models for a task using web search + LLM.

    Attributes:
        provider: The pluggable search provider to query.
        agent: Pydantic AI agent producing ``list[ModelCard]`` output.
        num_candidates: Number of candidate models to retrieve (default 4).
    """

    def __init__(
        self,
        provider: SearchProvider,
        model: str = "openai:gpt-4o",
        num_candidates: int = 4,
    ) -> None:
        """Create a retriever agent.

        Args:
            provider: Search provider used to gather evidence snippets.
            model: Pydantic AI model string.
            num_candidates: Desired number of candidate models (M).
        """
        self.provider = provider
        self.num_candidates = num_candidates
        self.agent = Agent(
            model,
            name="retriever_agent",
            output_type=list[ModelCard],
            instructions=_RETRIEVER_INSTRUCTIONS,
            defer_model_check=True,
        )

    def build_query(self, spec: TaskSpecification) -> str:
        """Build a targeted search query from the task specification.

        Args:
            spec: The validated task specification.

        Returns:
            A search query string combining task type, metric, and context.
        """
        parts = [
            spec.task_type.value,
            spec.metric_name,
            "state-of-the-art model",
            "python example",
        ]
        if spec.task_name:
            parts.insert(0, spec.task_name)
        return " ".join(parts)

    def retrieve(self, spec: TaskSpecification) -> RetrievedCandidates:
        """Retrieve structured candidate model cards for a task.

        Args:
            spec: The validated task specification.

        Returns:
            A ``RetrievedCandidates`` container holding the LLM-produced
            model cards, the query used, and the candidate count.
        """
        query = self.build_query(spec)
        with logfire.span("retriever.search", provider=self.provider.provider_name, query=query):
            results = self.provider.search(query, num_results=max(self.num_candidates, 5))

        search_context = "\n".join(f"- {r.title}: {r.url}\n  {r.snippet}" for r in results)
        prompt = (
            f"# Competition\n{spec.task_name or spec.task_type.value}\n"
            f"{spec.description or ''}\n"
            f"Evaluation metric: {spec.metric_name} "
            f"({spec.metric_direction.value})\n"
            f"# Web search results\n{search_context or '(no results)'}\n"
            f"# Your task\n"
            f"List {self.num_candidates} recent effective models and their "
            f"example codes to win the above competition."
        )
        with logfire.span("retriever.llm", num_candidates=self.num_candidates):
            result = self.agent.run_sync(prompt)
        cards = result.output
        return RetrievedCandidates(
            candidates=cards,
            query_used=query,
            total_found=len(cards),
        )
