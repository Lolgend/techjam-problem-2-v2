"""Search-guided retrieval subpackage.

Exposes the retriever agent and the pluggable search provider layer
(Tavily, Google Custom Search, DuckDuckGo, and the deterministic offline
mock).
"""

from problem_2_v2.search.providers import (
    DuckDuckGoSearchProvider,
    GoogleSearchProvider,
    MockSearchProvider,
    SearchProvider,
    SearchResult,
    TavilySearchProvider,
)
from problem_2_v2.search.retriever import RetrieverAgent

__all__ = [
    "DuckDuckGoSearchProvider",
    "GoogleSearchProvider",
    "MockSearchProvider",
    "RetrieverAgent",
    "SearchProvider",
    "SearchResult",
    "TavilySearchProvider",
]
