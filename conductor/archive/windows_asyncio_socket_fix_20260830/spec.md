# Specification: Fix Windows AsyncIO Socket Error (WinError 10038) and Thread Safety

## 1. Overview
Fixes the `OSError: [WinError 10038] An operation was attempted on something that is not a socket` exception occurring in `_ProactorBasePipeTransport._call_connection_lost` when running `problem-2-v2 run` on Windows platforms with concurrent parallel branches, and ensures thread-safe HTTP/socket operations in `DuckDuckGoSearchProvider`.

## 2. Functional Requirements

### A. Windows Event Loop Policy (`src/problem_2_v2/cli.py`, `src/problem_2_v2/orchestrator.py`)
- Detect Windows platform (`sys.platform == "win32"`).
- Globally configure `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` before initializing event loops to eliminate Proactor socket teardown crashes.

### B. Thread-Safe Search Provider (`src/problem_2_v2/search/providers.py`)
- Protect `DuckDuckGoSearchProvider.search()` with a `threading.Lock()`.
- Instantiate fresh, scoped `DDGS(timeout=20)` sessions per search call.
- Wrap search backend calls in `try...except Exception as exc:` to log warnings to Logfire and return `[]` on socket, network, or rate-limit failures rather than raising unhandled exceptions.

## 3. Non-Functional Requirements
- **Platform Resilience:** Works reliably across Windows, Linux, and macOS.
- **Graceful Degradation:** Temporary search engine glitches or rate-limits fallback cleanly without terminating the optimization run.
- **Backward Compatibility:** All existing 310 tests continue to pass 100% green.

## 4. Acceptance Criteria
- [ ] Setting `WindowsSelectorEventLoopPolicy` on Windows prevents `WinError 10038` during parallel branch execution.
- [ ] Concurrent `DuckDuckGoSearchProvider.search()` calls across parallel threads execute safely without race conditions.
- [ ] Search errors degrade gracefully to empty result lists without crashing the agent.
- [ ] Full test suite passes 100% green.
