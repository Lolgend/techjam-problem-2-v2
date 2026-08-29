"""Targeted refinement and ablation study contract schemas.

These models drive the inner loop of the MLE-STAR framework: they describe
isolated ablation experiments, aggregate their outcomes, identify the
highest-impact code block for optimization, and plan its refinement.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.contracts.code_utils import validate_python_syntax
from problem_2_v2.contracts.enums import ComponentCategory

__all__ = [
    "ComponentCategory",
    "AblationVariant",
    "AblationResultItem",
    "AblationReport",
    "TargetCodeBlock",
    "RefinementPlan",
    "block_in_script",
]


def block_in_script(code_block: str, script: str) -> bool:
    """Check whether a code block appears in a script (indentation-tolerant).

    The block may have been extracted without its surrounding indentation,
    so each line is matched against its stripped content at a line
    boundary.

    Args:
        code_block: The code block to locate.
        script: The script to search.

    Returns:
        True when every non-blank line of the block matches a run of
        script lines at line boundaries.
    """
    stripped_lines = [re.escape(line.strip()) for line in code_block.splitlines() if line.strip()]
    if not stripped_lines:
        return False
    pattern = re.compile(r"(?m)^[ \t]*" + r"\n[ \t]*".join(stripped_lines))
    return pattern.search(script) is not None


class AblationVariant(BaseModel):
    """A single isolated ablation experiment.

    Attributes:
        variant_id: Unique identifier for the variant.
        component_name: Name of the pipeline component being ablated.
        category: Functional category of the component.
        hypothesis: What the agent expects to learn from this ablation.
        modified_code_block: The isolated code block under test.
        ablation_code: Standalone script implementing the ablation.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    variant_id: str = Field(description="Unique variant identifier.")
    component_name: str = Field(description="Component under ablation.")
    category: ComponentCategory = Field(description="Component functional category.")
    hypothesis: str = Field(description="Stated hypothesis for the ablation.")
    modified_code_block: str = Field(description="The code block under test.")
    ablation_code: str = Field(description="Standalone ablation script.")


class AblationResultItem(BaseModel):
    """Outcome of a single ablation variant.

    Attributes:
        variant_id: Identifier of the executed ablation variant.
        validation_score: Validation score achieved by the variant.
        delta_from_baseline: Signed improvement over the baseline score.
        summary: Concise natural-language summary of the outcome.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    variant_id: str = Field(description="Executed variant identifier.")
    validation_score: float = Field(description="Achieved validation score.")
    delta_from_baseline: float = Field(description="Signed delta over the baseline.")
    summary: str = Field(description="Natural-language outcome summary.")


class AblationReport(BaseModel):
    """Aggregated report over all ablation variants.

    Attributes:
        baseline_score: The reference validation score.
        ablation_results: Per-variant outcomes.
        highest_impact_component: Name of the component with the largest
            positive delta.
        raw_log_summary: Raw execution output summary for the log.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    baseline_score: float = Field(description="Reference baseline score.")
    ablation_results: list[AblationResultItem] = Field(
        default_factory=list,
        description="Per-variant outcomes.",
    )
    highest_impact_component: str = Field(description="Highest-impact component name.")
    raw_log_summary: str = Field(description="Raw log summary.")

    def highest_impact_result(self) -> AblationResultItem | None:
        """Return the ablation result with the largest positive delta.

        Returns:
            The highest-impact result, or ``None`` when there are no
            results to compare.
        """
        if not self.ablation_results:
            return None
        return max(self.ablation_results, key=lambda item: item.delta_from_baseline)


class TargetCodeBlock(BaseModel):
    """An extracted code segment targeted for optimization.

    Attributes:
        raw_code: The extracted source of the code block.
        category: Functional category of the block.
        start_line: 1-indexed start line in the full script, when known.
        end_line: 1-indexed end line in the full script, when known.
        initial_plan: Draft improvement plan for the block.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    raw_code: str = Field(description="Extracted block source.")
    category: ComponentCategory = Field(description="Block functional category.")
    start_line: int | None = Field(default=None, description="1-indexed start line.")
    end_line: int | None = Field(default=None, description="1-indexed end line.")
    initial_plan: str = Field(description="Draft improvement plan.")

    def replace_in(self, full_script: str, new_code: str) -> str:
        """Replace this block in a full script and validate the result.

        When ``start_line``/``end_line`` are known the block is located by
        line numbers; otherwise the ``raw_code`` substring is located.
        The resulting script must parse as valid Python.

        Args:
            full_script: The complete solution script.
            new_code: The replacement code block.

        Returns:
            The full script with the block replaced.

        Raises:
            ValueError: If the block cannot be located, the line range is
                invalid, or the replacement produces invalid Python.
        """
        if self.start_line is not None and self.end_line is not None:
            candidate = self._replace_by_lines(full_script, new_code)
        else:
            candidate = self._replace_by_substring(full_script, new_code)

        valid, error = validate_python_syntax(candidate)
        if not valid:
            raise ValueError(f"Replacement produced invalid Python: {error}")
        return candidate

    def _replace_by_substring(self, full_script: str, new_code: str) -> str:
        pattern = self._build_pattern()
        match = pattern.search(full_script)
        if match is None:
            raise ValueError("Target code block not found in full script.")
        matched_text = match.group(0)
        first_line = matched_text.splitlines()[0]
        indent = first_line[: len(first_line) - len(first_line.lstrip())]
        indented_new = "".join(f"{indent}{line}\n" for line in new_code.splitlines())
        tail = full_script[match.end() :]
        if tail.startswith("\n"):
            tail = tail[1:]
        return full_script[: match.start()] + indented_new + tail

    def _build_pattern(self) -> re.Pattern[str]:
        """Build a line-anchored, indentation-tolerant block pattern.

        The raw block may have been extracted without its surrounding
        indentation, so each line is matched against its stripped content
        at a line boundary.
        """
        stripped_lines = [
            re.escape(line.strip()) for line in self.raw_code.splitlines() if line.strip()
        ]
        if not stripped_lines:
            raise ValueError("Target code block is empty; cannot locate it in the full script.")
        return re.compile(r"(?m)^[ \t]*" + r"\n[ \t]*".join(stripped_lines))

    def _replace_by_lines(self, full_script: str, new_code: str) -> str:
        start_line = self.start_line
        end_line = self.end_line
        if start_line is None or end_line is None:
            raise ValueError("start_line and end_line must both be set for line-based replacement.")
        lines = full_script.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError(
                f"Invalid line range [{start_line}, {end_line}] for script with {len(lines)} lines."
            )
        if new_code and not new_code.endswith("\n"):
            new_code = f"{new_code}\n"
        replaced = lines[: start_line - 1] + [new_code] + lines[end_line:]
        return "".join(replaced)


class RefinementPlan(BaseModel):
    """Inner-loop refinement planning model.

    Attributes:
        plan_id: Unique plan identifier.
        natural_language_plan: The proposed refinement strategy in prose.
        target_subcomponents: Sub-components the plan will modify.
        expected_gain: Anticipated score gain in prose form.
        iteration_index: Inner-loop iteration this plan belongs to.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    plan_id: str = Field(description="Unique plan identifier.")
    natural_language_plan: str = Field(description="Refinement strategy in prose.")
    target_subcomponents: list[str] = Field(
        default_factory=list,
        description="Sub-components targeted by the plan.",
    )
    expected_gain: str = Field(description="Anticipated score gain.")
    iteration_index: int = Field(description="Inner-loop iteration index.")
