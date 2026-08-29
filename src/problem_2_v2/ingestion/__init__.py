"""Task ingestion subpackage.

Exposes the task extractor that parses markdown problem descriptions into
validated ``TaskSpecification`` instances.
"""

from problem_2_v2.ingestion.extractor import TaskExtractor

__all__ = ["TaskExtractor"]
