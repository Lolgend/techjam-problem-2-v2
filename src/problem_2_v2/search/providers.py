"""Pluggable web search provider layer.

Defines the ``SearchProvider`` protocol and four implementations:
Tavily API, Google Custom Search JSON API, DuckDuckGo (free fallback), and
a deterministic offline mock for testing.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import httpx
import logfire
from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """A single web search hit.

    Attributes:
        title: Result title.
        url: Result URL.
        snippet: Short text excerpt describing the result.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    title: str = Field(description="Result title.")
    url: str = Field(description="Result URL.")
    snippet: str = Field(default="", description="Short excerpt.")


@runtime_checkable
class SearchProvider(Protocol):
    """Interface every search provider must implement.

    Implementations are synchronous and should be safe to call from the
    agent runtime.
    """

    provider_name: str

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute a web search.

        Args:
            query: The search query.
            num_results: Maximum number of results to return.

        Returns:
            The ranked list of search results (may be empty).
        """
        ...


class MockSearchProvider:
    """Deterministic offline search provider for unit and integration tests.

    Results are looked up by substring keyword matching over the query.
    """

    def __init__(self, results: dict[str, list[SearchResult]] | None = None) -> None:
        """Create a mock provider.

        Args:
            results: Mapping from keyword substrings to canned results.
        """
        self._results = results or {}
        self.provider_name = "mock"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Return canned results whose keyword appears in the query."""
        query_lower = query.lower()
        matched: list[SearchResult] = []
        for keyword, hits in self._results.items():
            if keyword.lower() in query_lower:
                matched.extend(hits)
        return matched[:num_results]


class TavilySearchProvider:
    """Search provider backed by the Tavily Search API.

    Args:
        api_key: Tavily API key; falls back to the ``TAVILY_API_KEY``
            environment variable.
        client: Optional ``httpx.Client`` for injection in tests.
    """

    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self._api_key:
            raise ValueError("TavilySearchProvider requires an API key (TAVILY_API_KEY).")
        self._client = client or httpx.Client(timeout=30.0)
        self.provider_name = "tavily"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Query the Tavily API and map responses to search results."""
        with logfire.span("search.tavily", query=query, num_results=num_results):
            response = self._client.post(
                self._ENDPOINT,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": num_results,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in payload.get("results", [])
        ]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> TavilySearchProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class GoogleSearchProvider:
    """Search provider backed by the Google Custom Search JSON API.

    Args:
        api_key: Google API key; falls back to ``GOOGLE_API_KEY``.
        cx: Custom Search Engine id; falls back to ``GOOGLE_CSE_ID``.
        client: Optional ``httpx.Client`` for injection in tests.
    """

    _ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str | None = None,
        cx: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._cx = cx or os.environ.get("GOOGLE_CSE_ID", "")
        if not self._api_key:
            raise ValueError("GoogleSearchProvider requires an API key (GOOGLE_API_KEY).")
        if not self._cx:
            raise ValueError(
                "GoogleSearchProvider requires a custom search engine id (cx / GOOGLE_CSE_ID)."
            )
        self._client = client or httpx.Client(timeout=30.0)
        self.provider_name = "google"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Query the Google Custom Search API and map items to results."""
        with logfire.span("search.google", query=query, num_results=num_results):
            response = self._client.get(
                self._ENDPOINT,
                params={
                    "key": self._api_key,
                    "cx": self._cx,
                    "q": query,
                    "num": num_results,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in payload.get("items", [])
        ]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> GoogleSearchProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DuckDuckGoSearchProvider:
    """Free web search provider backed by the ``ddgs`` package.

    Args:
        backend: A DuckDuckGo ``DDGS``-compatible backend; when omitted a
            real ``DDGS`` instance is created. Tests inject a fake.
    """

    def __init__(self, backend: Any | None = None) -> None:
        if backend is None:
            from ddgs import DDGS

            backend = DDGS()
        self._backend = backend
        self.provider_name = "duckduckgo"

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Query DuckDuckGo and map the raw dicts to search results."""
        with logfire.span("search.duckduckgo", query=query, num_results=num_results):
            raw = self._backend.text(query, max_results=num_results)
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
            )
            for item in raw
        ]
