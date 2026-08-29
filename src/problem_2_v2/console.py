"""Console telemetry helpers for live interactive progress output.

All announcements flush immediately so terminal emulators render output
with zero delay, and score/delta formatters render ``None`` gracefully.
"""

from __future__ import annotations


def announce(message: str) -> None:
    """Print a telemetry message with an immediate flush.

    Args:
        message: The line to render on the console.
    """
    print(message, flush=True)


def format_score(value: float | None, digits: int = 4) -> str:
    """Format a validation score, rendering ``None`` as ``n/a``.

    Args:
        value: The score to format.
        digits: Number of decimal places.

    Returns:
        The formatted score string.
    """
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_delta(value: float | None, digits: int = 4) -> str:
    """Format a signed score delta, rendering ``None`` as ``n/a``.

    Args:
        value: The delta to format.
        digits: Number of decimal places.

    Returns:
        The formatted delta string, signed.
    """
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"
