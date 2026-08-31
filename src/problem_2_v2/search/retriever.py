"""Retriever agent ($A_retriever$): search-guided candidate model retrieval.

Formulates a targeted query from the task specification, fetches web
results through a pluggable ``SearchProvider``, and prompts the LLM
(Figure 9 prompt) to distill the search context into structured model
cards. Retrieval is resilient: structured output is the primary path, raw
JSON/markdown text parsing is the fallback, and domain-aware starter
architectures guarantee a non-empty candidate list.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.enums import TaskType
from problem_2_v2.contracts.search import ModelCard, RetrievedCandidates
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.search.providers import SearchProvider

_RETRIEVER_INSTRUCTIONS = (
    "You are retrieving state-of-the-art models for a machine learning competition.\n"
    "You have access to the `search_web` tool to search the internet for competition "
    "winning solutions, SOTA architectures, GitHub repositories, and Python example codes.\n"
    "Search the web using `search_web` if you need more information or evidence before answering.\n"
)

_RETRIEVER_PROMPT_TEMPLATE = (
    "# Competition\n"
    "{task_description}\n\n"
    "# Your task\n"
    "- List {num_candidates} recent effective models and their example codes to win the above competition.\n\n"
    "# Requirement\n"
    "- The example code should be concise and simple.\n"
    "- You must provide an example code, i.e., do not just mention GitHubs or papers.\n\n"
    "# OUTPUT JSON schema:\n"
    "- Model = {{'model_name': str, 'example_code': str}}\n"
    "- Return: list[Model]\n"
)


_FENCE_RE = re.compile(r"```(?:json|python)?\s*\n?(.*?)```", re.DOTALL)
_BULLET_RE = re.compile(
    r"^[-*]\s+\*{0,2}(?P<name>[A-Za-z0-9][^*]*?)\*{0,2}\s*(?::\s*(?P<detail>.*))?$"
)

_MODEL_FIELDS = frozenset(ModelCard.model_fields)


class RetrieverAgent:
    """Retrieves candidate models for a task using autonomous web search tool + LLM.

    Attributes:
        provider: The pluggable search provider queried by the search tool.
        agent: Pydantic AI agent producing ``list[ModelCard]`` output.
        text_agent: Pydantic AI agent returning raw text for fallback
            parsing when structured tool-calling is unavailable.
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
        search_tool = self._build_search_tool()

        self.agent = Agent(
            model,
            name="retriever_agent",
            output_type=list[ModelCard],
            instructions=_RETRIEVER_INSTRUCTIONS,
            tools=[search_tool],
            defer_model_check=True,
        )
        self.text_agent = Agent(
            model,
            name="retriever_text_agent",
            output_type=str,
            instructions=_RETRIEVER_INSTRUCTIONS,
            tools=[search_tool],
            defer_model_check=True,
        )

    def _build_search_tool(self):
        """Create a callable search_web function tool bound to the search provider."""
        provider = self.provider

        def search_web(query: str, num_results: int = 5) -> str:
            """Search the web for machine learning models, architectures, or code examples.

            Args:
                query: The search query terms.
                num_results: Max number of search results to return (default 5).

            Returns:
                A formatted string of search results or an informative error message.
            """
            with logfire.span("retriever.search_tool", provider=provider.provider_name, query=query):
                try:
                    results = provider.search(query, num_results=num_results)
                except Exception as exc:
                    return f"Search failed: {exc}"

                if not results:
                    return "No search results found."

                return "\n".join(f"- {r.title} ({r.url}):\n  {r.snippet}" for r in results)

        return search_web

    def build_prompt(self, spec: TaskSpecification) -> str:
        """Build the user prompt for the retriever agent.

        Args:
            spec: The validated task specification.

        Returns:
            The formatted prompt string.
        """
        task_desc = (
            f"{spec.task_name or spec.task_type.value}\n"
            f"{spec.description or ''}\n"
            f"Evaluation metric: {spec.metric_name} ({spec.metric_direction.value})"
        ).strip()
        return _RETRIEVER_PROMPT_TEMPLATE.format(
            task_description=task_desc,
            num_candidates=self.num_candidates,
        )

    def retrieve(self, spec: TaskSpecification) -> RetrievedCandidates:
        """Retrieve structured candidate model cards for a task.

        Structured output is attempted first; when the LLM returns empty or
        raises, the raw text response is parsed for JSON/markdown cards;
        if that also fails, domain-aware fallback architectures are
        returned so the candidate list is never empty.

        Args:
            spec: The validated task specification.

        Returns:
            A ``RetrievedCandidates`` container holding the candidate model
            cards and the candidate count.
        """
        prompt = self.build_prompt(spec)

        cards: list[ModelCard] = []
        with logfire.span("retriever.llm", num_candidates=self.num_candidates):
            try:
                result = self.agent.run_sync(prompt)
                cards = list(result.output or [])
            except Exception as exc:
                logfire.warn("retriever.structured_failed", error=str(exc))

        if not cards:
            cards = self._text_fallback(prompt)

        if not cards:
            cards = self.get_domain_fallback_cards(spec.task_type)

            logfire.warn(
                "retriever.using_domain_fallback",
                task_type=spec.task_type.value,
                count=len(cards),
            )

        return RetrievedCandidates(
            candidates=cards,
            total_found=len(cards),
        )


    def _text_fallback(self, prompt: str) -> list[ModelCard]:
        """Parse model cards from the raw text response."""
        try:
            response = self.text_agent.run_sync(prompt)
            cards = self._parse_cards(response.output)
        except Exception as exc:
            logfire.warn("retriever.text_fallback_failed", error=str(exc))
            return []
        if cards:
            logfire.warn(
                "retriever.text_fallback_used",
                parsed=len(cards),
            )
        return cards

    @staticmethod
    def _parse_cards(raw_text: str) -> list[ModelCard]:
        """Parse ``ModelCard`` instances from raw JSON or markdown text.

        Tries a raw JSON array first, then a JSON array inside fenced code
        blocks, then a markdown bullet/code-block list.

        Args:
            raw_text: The raw LLM response text.

        Returns:
            The parsed model cards (possibly empty).
        """
        if not raw_text or not raw_text.strip():
            return []

        parsed = RetrieverAgent._try_load_json(raw_text)
        if parsed is not None:
            return RetrieverAgent._cards_from_json(parsed)

        for match in _FENCE_RE.finditer(raw_text):
            parsed = RetrieverAgent._try_load_json(match.group(1))
            if parsed is not None:
                return RetrieverAgent._cards_from_json(parsed)

        return RetrieverAgent._parse_markdown_cards(raw_text)

    @staticmethod
    def _try_load_json(text: str) -> list[dict[str, Any]] | None:
        """Load a JSON array from text, or return ``None``."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return None

    @staticmethod
    def _cards_from_json(items: list[dict[str, Any]]) -> list[ModelCard]:
        """Build ``ModelCard`` instances from JSON dicts (unknown keys dropped)."""
        cards: list[ModelCard] = []
        for item in items:
            filtered = {key: value for key, value in item.items() if key in _MODEL_FIELDS}
            if "model_name" in filtered and "example_code" in filtered:
                cards.append(ModelCard(**filtered))
        return cards

    @staticmethod
    def _parse_markdown_cards(raw_text: str) -> list[ModelCard]:
        """Parse model cards from a markdown bullet + code-block list."""
        lines = raw_text.splitlines()
        cards: list[ModelCard] = []
        name: str | None = None
        rationale: list[str] = []
        code: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index].strip()
            bullet = _BULLET_RE.match(line)
            if bullet:
                if name is not None:
                    cards.append(RetrieverAgent._build_markdown_card(name, rationale, code))
                name = bullet.group("name").strip()
                detail = bullet.group("detail")
                rationale = [detail.strip()] if detail and detail.strip() else []
                code = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    if lines[index].strip():
                        rationale.append(lines[index].strip())
                    index += 1
                if index < len(lines):
                    index += 1  # skip opening fence
                    while index < len(lines) and not lines[index].strip().startswith("```"):
                        code.append(lines[index])
                        index += 1
                    index += 1  # skip closing fence
                continue
            index += 1

        if name is not None:
            cards.append(RetrieverAgent._build_markdown_card(name, rationale, code))
        return cards

    @staticmethod
    def _build_markdown_card(
        name: str,
        rationale: list[str],
        code: list[str],
    ) -> ModelCard:
        """Build a ``ModelCard`` from parsed markdown parts."""
        example_code = textwrap.dedent("\n".join(code)).strip()
        return ModelCard(
            model_name=name,
            rationale=" ".join(rationale).strip() or "Retrieved candidate.",
            example_code=example_code,
        )

    @staticmethod
    def get_domain_fallback_cards(task_type: TaskType) -> list[ModelCard]:
        """Return domain-aware starter architectures for a task type.

        Guarantees a non-empty candidate list even when search and LLM
        retrieval both fail.

        Args:
            task_type: The machine learning task family.

        Returns:
            Starter ``ModelCard`` instances for the domain.
        """
        return _DOMAIN_FALLBACKS.get(task_type, _DOMAIN_FALLBACKS[TaskType.TABULAR_CLASSIFICATION])


def _card(name: str, rationale: str, code: str, deps: list[str] | None = None) -> ModelCard:
    """Build a starter model card."""
    return ModelCard(
        model_name=name,
        rationale=rationale,
        example_code=code,
        library_dependencies=deps or [],
    )


_DOMAIN_FALLBACKS: dict[TaskType, list[ModelCard]] = {
    TaskType.RECOMMENDER_RANKING: [
        _card(
            "Factorization Machine (BPR Loss)",
            "Pairwise FM ranking model for implicit feedback.",
            "import numpy as np\n# FM with BPR pairwise loss placeholder",
            ["numpy"],
        ),
        _card(
            "DeepFM",
            "Deep Factorization Machine combining FM and deep network.",
            "import torch\n# DeepFM: FM layer + deep MLP tower",
            ["torch"],
        ),
        _card(
            "LightGBM Ranker",
            "Gradient boosted ranking with LambdaRank.",
            "import lightgbm as lgb\nranker = lgb.LGBMRanker()",
            ["lightgbm"],
        ),
        _card(
            "Item Popularity",
            "Non-personalized item popularity baseline.",
            "from collections import Counter\n# rank by item popularity",
            [],
        ),
    ],
    TaskType.TABULAR_CLASSIFICATION: [
        _card(
            "LightGBM",
            "Gradient boosting for tabular classification.",
            "import lightgbm as lgb\nclf = lgb.LGBMClassifier()",
            ["lightgbm"],
        ),
        _card(
            "XGBoost",
            "Regularized gradient boosting for tabular data.",
            "import xgboost as xgb\nclf = xgb.XGBClassifier()",
            ["xgboost"],
        ),
        _card(
            "CatBoost",
            "Categorical gradient boosting.",
            "import catboost as cb\nclf = cb.CatBoostClassifier(verbose=False)",
            ["catboost"],
        ),
        _card(
            "Multi-Layer Perceptron",
            "Feed-forward neural network baseline.",
            "from sklearn.neural_network import MLPClassifier\nclf = MLPClassifier()",
            ["scikit-learn"],
        ),
    ],
    TaskType.TABULAR_REGRESSION: [
        _card(
            "LightGBM Regressor",
            "Gradient boosting for tabular regression.",
            "import lightgbm as lgb\nreg = lgb.LGBMRegressor()",
            ["lightgbm"],
        ),
        _card(
            "XGBoost Regressor",
            "Regularized gradient boosting regression.",
            "import xgboost as xgb\nreg = xgb.XGBRegressor()",
            ["xgboost"],
        ),
        _card(
            "CatBoost Regressor",
            "Categorical gradient boosting regression.",
            "import catboost as cb\nreg = cb.CatBoostRegressor(verbose=False)",
            ["catboost"],
        ),
        _card(
            "Multi-Layer Perceptron Regressor",
            "Feed-forward neural network regression baseline.",
            "from sklearn.neural_network import MLPRegressor\nreg = MLPRegressor()",
            ["scikit-learn"],
        ),
    ],
    TaskType.IMAGE_CLASSIFICATION: [
        _card(
            "ResNet Baseline",
            "ImageNet-pretrained residual network.",
            "import torchvision.models as models\nmodel = models.resnet18(weights=None)",
            ["torch", "torchvision"],
        ),
        _card(
            "EfficientNet Baseline",
            "Scalable convolutional network.",
            "import torchvision.models as models\nmodel = models.efficientnet_b0(weights=None)",
            ["torch", "torchvision"],
        ),
    ],
    TaskType.IMAGE_TO_IMAGE: [
        _card(
            "UNet Baseline",
            "Encoder-decoder segmentation architecture.",
            "import torch.nn as nn\n# UNet encoder-decoder placeholder",
            ["torch"],
        ),
        _card(
            "Autoencoder Baseline",
            "Reconstruction autoencoder.",
            "import torch.nn as nn\n# convolutional autoencoder placeholder",
            ["torch"],
        ),
    ],
    TaskType.TEXT_CLASSIFICATION: [
        _card(
            "Transformer Classifier",
            "Fine-tuned transformer encoder for text.",
            (
                "from transformers import AutoModelForSequenceClassification\n"
                "model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')"
            ),
            ["transformers"],
        ),
        _card(
            "TF-IDF + Logistic Regression",
            "Bag-of-words linear baseline.",
            (
                "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                "from sklearn.linear_model import LogisticRegression"
            ),
            ["scikit-learn"],
        ),
    ],
    TaskType.SEQ_TO_SEQ: [
        _card(
            "Seq2Seq Transformer",
            "Encoder-decoder transformer for sequence generation.",
            (
                "from transformers import AutoModelForSeq2SeqLM\n"
                "model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')"
            ),
            ["transformers"],
        ),
    ],
    TaskType.AUDIO_CLASSIFICATION: [
        _card(
            "CNN Spectrogram Baseline",
            "Convolutional network over log-mel spectrograms.",
            "import torch.nn as nn\n# CNN over spectrogram placeholder",
            ["torch"],
        ),
    ],
    TaskType.MULTIMODAL: [
        _card(
            "Late-Fusion Model",
            "Late fusion of per-modality encoders.",
            "import torch.nn as nn\n# per-modality encoders + fusion head placeholder",
            ["torch"],
        ),
    ],
}
