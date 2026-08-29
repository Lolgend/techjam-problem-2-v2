"""Unit tests for the pluggable search provider layer."""

import httpx
import pytest
from httpx import Response

from problem_2_v2.search.providers import (
    DuckDuckGoSearchProvider,
    GoogleSearchProvider,
    MockSearchProvider,
    SearchResult,
    TavilySearchProvider,
)


class FakeClient:
    """Minimal httpx.Client stand-in for offline provider tests."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object] | None = None, **_: object) -> Response:
        self.requests.append({"method": "post", "url": url, "json": json})
        request = httpx.Request("POST", url, json=json)
        return Response(200, json=self._tavily_response(), request=request)

    def get(self, url: str, params: dict[str, object] | None = None, **_: object) -> Response:
        self.requests.append({"method": "get", "url": url, "params": params})
        request = httpx.Request("GET", url, params=params)
        return Response(200, json=self._google_response(), request=request)

    @staticmethod
    def _tavily_response() -> dict[str, object]:
        return {
            "results": [
                {"title": "Tavily Hit", "url": "https://example.com/tavily", "content": "snippet"},
            ]
        }

    @staticmethod
    def _google_response() -> dict[str, object]:
        return {
            "items": [
                {"title": "Google Hit", "link": "https://example.com/google", "snippet": "snippet"},
            ]
        }


class TestSearchResult:
    """Test the `SearchResult` model."""

    def test_instantiates(self) -> None:
        result = SearchResult(title="t", url="https://x.com", snippet="s")
        assert result.title == "t"
        assert result.url == "https://x.com"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            SearchResult(title="t", url="https://x.com", snippet="s", extra=1)  # type: ignore[call-arg]


class TestMockSearchProvider:
    """Test the deterministic offline provider."""

    def test_returns_configured_results_for_keyword(self) -> None:
        hit = SearchResult(title="LightGBM CTR", url="https://example.com/lgbm", snippet="...")
        provider = MockSearchProvider(results={"ctr": [hit]})
        results = provider.search("ctr prediction model")
        assert results == [hit]

    def test_returns_empty_for_unknown_query(self) -> None:
        provider = MockSearchProvider(results={})
        assert provider.search("no such keyword") == []

    def test_honors_num_results_limit(self) -> None:
        hits = [SearchResult(title=f"m{i}", url=f"https://e.com/{i}", snippet="") for i in range(5)]
        provider = MockSearchProvider(results={"model": hits})
        assert len(provider.search("best model", num_results=3)) == 3

    def test_is_deterministic(self) -> None:
        hit = SearchResult(title="A", url="https://a.com", snippet="")
        provider = MockSearchProvider(results={"a": [hit, hit]})
        assert provider.search("a b c") == provider.search("a b c")

    def test_provider_name(self) -> None:
        assert MockSearchProvider(results={}).provider_name == "mock"


class TestTavilySearchProvider:
    """Test the Tavily REST integration with an injected client."""

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="(?i)api key"):
            TavilySearchProvider(api_key="")  # type: ignore[arg-type]

    def test_posts_query_and_parses_results(self) -> None:
        client = FakeClient()
        provider = TavilySearchProvider(api_key="secret", client=client)  # type: ignore[arg-type]
        results = provider.search("ctr model", num_results=5)
        assert results[0].title == "Tavily Hit"
        assert results[0].url == "https://example.com/tavily"
        request = client.requests[0]
        assert request["url"] == "https://api.tavily.com/search"
        payload = request["json"]
        assert payload is not None
        assert payload["api_key"] == "secret"
        assert payload["query"] == "ctr model"
        assert payload["max_results"] == 5

    def test_empty_results_on_empty_response(self) -> None:
        class EmptyClient(FakeClient):
            @staticmethod
            def _tavily_response() -> dict[str, object]:
                return {"results": []}

        provider = TavilySearchProvider(api_key="k", client=EmptyClient())  # type: ignore[arg-type]
        assert provider.search("anything") == []

    def test_provider_name(self) -> None:
        provider = TavilySearchProvider(api_key="k", client=FakeClient())  # type: ignore[arg-type]
        assert provider.provider_name == "tavily"

    def test_context_manager_closes_client(self) -> None:
        class TrackedClient(FakeClient):
            closed = False

            def close(self) -> None:
                TrackedClient.closed = True

        with TavilySearchProvider(api_key="k", client=TrackedClient()) as provider:  # type: ignore[arg-type]
            assert provider.search("q")
        assert TrackedClient.closed is True


class TestGoogleSearchProvider:
    """Test the Google Custom Search integration with an injected client."""

    def test_requires_api_key_and_cx(self) -> None:
        with pytest.raises(ValueError, match="(?i)api key"):
            GoogleSearchProvider(api_key="", cx="abc")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="(?i)cx"):
            GoogleSearchProvider(api_key="k", cx="")

    def test_gets_query_and_parses_items(self) -> None:
        client = FakeClient()
        provider = GoogleSearchProvider(api_key="k", cx="cx1", client=client)  # type: ignore[arg-type]
        results = provider.search("click-through ranking", num_results=5)
        assert results[0].title == "Google Hit"
        assert results[0].snippet == "snippet"
        request = client.requests[0]
        params = request["params"]
        assert params is not None
        assert params["key"] == "k"
        assert params["cx"] == "cx1"
        assert params["q"] == "click-through ranking"
        assert params["num"] == 5

    def test_provider_name(self) -> None:
        provider = GoogleSearchProvider(api_key="k", cx="c", client=FakeClient())  # type: ignore[arg-type]
        assert provider.provider_name == "google"


class TestDuckDuckGoSearchProvider:
    """Test the DuckDuckGo provider with an injected search backend."""

    def test_queries_backend_and_maps_results(self) -> None:
        class FakeDDGS:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def text(
                self, keywords: str, max_results: int | None = None, **_: object
            ) -> list[dict[str, str]]:
                self.calls.append({"keywords": keywords, "max_results": max_results})
                return [{"title": "DDG Hit", "href": "https://example.com/ddg", "body": "snippet"}]

        fake = FakeDDGS()
        provider = DuckDuckGoSearchProvider(backend=fake)  # type: ignore[arg-type]
        results = provider.search("vision model", num_results=7)
        assert results[0].title == "DDG Hit"
        assert fake.calls[0]["keywords"] == "vision model"
        assert fake.calls[0]["max_results"] == 7

    def test_empty_results_on_no_hits(self) -> None:
        class EmptyDDGS:
            def text(
                self, keywords: str, max_results: int | None = None, **_: object
            ) -> list[dict[str, str]]:
                return []

        provider = DuckDuckGoSearchProvider(backend=EmptyDDGS())  # type: ignore[arg-type]
        assert provider.search("nothing") == []

    def test_provider_name(self) -> None:
        class EmptyDDGS:
            def text(
                self, keywords: str, max_results: int | None = None, **_: object
            ) -> list[dict[str, str]]:
                return []

        provider = DuckDuckGoSearchProvider(backend=EmptyDDGS())  # type: ignore[arg-type]
        assert provider.provider_name == "duckduckgo"
