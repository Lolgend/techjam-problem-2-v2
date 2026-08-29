"""Windows AsyncIO compatibility and thread-safe search provider tests.

Verifies the ``WindowsSelectorEventLoopPolicy`` is configured on Windows to
prevent the Proactor ``WinError 10038`` socket teardown crash, and that
``DuckDuckGoSearchProvider`` serializes concurrent searches, degrades
gracefully on network/socket errors, and scopes fresh backend sessions.
"""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from problem_2_v2.orchestrator import configure_event_loop_policy
from problem_2_v2.search.providers import DuckDuckGoSearchProvider


class TestWindowsEventLoopPolicy:
    """Test the Windows selector event loop policy configuration."""

    def test_selector_policy_configured_on_windows(self) -> None:
        previous = asyncio.get_event_loop_policy()
        try:
            configure_event_loop_policy()
            policy = asyncio.get_event_loop_policy()
            if sys.platform == "win32":
                assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)
        finally:
            asyncio.set_event_loop_policy(previous)

    def test_policy_configuration_is_idempotent(self) -> None:
        previous = asyncio.get_event_loop_policy()
        try:
            configure_event_loop_policy()
            configure_event_loop_policy()
        finally:
            asyncio.set_event_loop_policy(previous)


class TestThreadSafeSearchProvider:
    """Test concurrent DuckDuckGo searches and graceful degradation."""

    def test_concurrent_searches_serialize_safely(self) -> None:
        class FakeDDGS:
            def __init__(self) -> None:
                self.calls = 0

            def text(self, keywords, max_results=None, **_):
                current = self.calls
                time.sleep(0.001)
                self.calls = current + 1
                return [{"title": f"Hit {self.calls}", "href": "https://e.com", "body": "s"}]

        backend = FakeDDGS()
        provider = DuckDuckGoSearchProvider(backend=backend)

        def run_search(_: int) -> list[object]:
            return provider.search("recommendation model")

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_search, range(16)))
        assert len(results) == 16
        assert all(len(r) == 1 for r in results)
        assert backend.calls == 16

    def test_search_error_degrades_to_empty(self) -> None:
        class BrokenDDGS:
            def text(self, keywords, max_results=None, **_):
                raise OSError("An operation was attempted on something that is not a socket")

        provider = DuckDuckGoSearchProvider(backend=BrokenDDGS())
        assert provider.search("query") == []

    def test_scoped_backend_session_per_call(self, monkeypatch) -> None:
        instances = {"count": 0}

        class FakeDDGS:
            def __init__(self, timeout: int = 20) -> None:
                instances["count"] += 1

            def text(self, keywords, max_results=None, **_):
                return [{"title": "t", "href": "https://e.com", "body": "b"}]

        monkeypatch.setattr("problem_2_v2.search.providers.DDGS", FakeDDGS)
        provider = DuckDuckGoSearchProvider()
        assert len(provider.search("q")) == 1
        assert len(provider.search("q")) == 1
        assert instances["count"] == 2
