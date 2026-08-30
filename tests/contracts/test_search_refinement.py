"""Unit tests for search retrieval, ablation, and refinement schemas."""

import pytest
from pydantic import ValidationError

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import (
    AblationReport,
    AblationResultItem,
    AblationVariant,
    RefinementPlan,
    TargetCodeBlock,
    align_replacement_indent,
    find_matching_block,
)
from problem_2_v2.contracts.search import ModelCard, RetrievedCandidates


class TestFindMatchingBlock:
    """Test multi-tier resilient code block location."""

    SCRIPT = (
        "import pandas as pd\n"
        "def train_model():\n"
        "    X = data.drop(columns=['y'])\n"
        "    y = data['y']\n"
        "    model = LogisticRegression()\n"
        "    model.fit(X, y)\n"
        "    return model\n"
    )

    def test_verbatim_block_matches(self) -> None:
        block = "    X = data.drop(columns=['y'])\n    y = data['y']"
        expected = "    X = data.drop(columns=['y'])\n    y = data['y']"
        assert find_matching_block(block, self.SCRIPT) == expected

    def test_indentation_tolerant_verbatim_match(self) -> None:
        block = "X = data.drop(columns=['y'])\ny = data['y']"
        expected = "    X = data.drop(columns=['y'])\n    y = data['y']"
        assert find_matching_block(block, self.SCRIPT) == expected

    def test_quote_style_difference_matches_verbatim(self) -> None:
        block = 'X = data.drop(columns=["y"])\ny = data["y"]'
        expected = "    X = data.drop(columns=['y'])\n    y = data['y']"
        assert find_matching_block(block, self.SCRIPT) == expected

    def test_trailing_comment_difference_matches_verbatim(self) -> None:
        script = "model = LogisticRegression()  # baseline\nmodel.fit(X, y)\n"
        block = "model = LogisticRegression()\nmodel.fit(X, y)"
        assert find_matching_block(block, script) == (
            "model = LogisticRegression()  # baseline\nmodel.fit(X, y)"
        )

    def test_contiguous_anchor_subset_matches(self) -> None:
        block = (
            "X = data.drop(columns=['y'])\n"
            "y = data['y']\n"
            "model = LinearRegression()\n"
            "model.fit(X, y)"
        )
        assert find_matching_block(block, self.SCRIPT) == (
            "    X = data.drop(columns=['y'])\n    y = data['y']"
        )

    def test_ast_definition_fallback_matches(self) -> None:
        block = (
            "def train_model():\n"
            '    model = LogisticRegression(penalty="l2")\n'
            "    model.fit(X, y)\n"
            "    return model"
        )
        assert find_matching_block(block, self.SCRIPT) == (
            "def train_model():\n"
            "    X = data.drop(columns=['y'])\n"
            "    y = data['y']\n"
            "    model = LogisticRegression()\n"
            "    model.fit(X, y)\n"
            "    return model"
        )

    def test_returns_none_when_no_match(self) -> None:
        assert find_matching_block("totally = unrelated", self.SCRIPT) is None

    def test_returns_none_on_empty_block(self) -> None:
        assert find_matching_block("   \n  ", self.SCRIPT) is None


class TestAlignReplacementIndent:
    """Test indentation normalization for replacement code blocks."""

    def test_applies_base_indent_to_unindented_replacement(self) -> None:
        aligned = align_replacement_indent("model = XGB()\nmodel.fit(X, y)", "    ")
        assert aligned == "    model = XGB()\n    model.fit(X, y)"

    def test_normalizes_preindented_replacement_without_doubling(self) -> None:
        aligned = align_replacement_indent("    model = XGB()\n    model.fit(X, y)", "    ")
        assert aligned == "    model = XGB()\n    model.fit(X, y)"

    def test_preserves_relative_nesting_of_mixed_indent(self) -> None:
        replacement = (
            "    model = XGB()\n"
            "    if use_gpu:\n"
            "        model = XGB(tree_method='gpu_hist')\n"
            "    model.fit(X, y)"
        )
        aligned = align_replacement_indent(replacement, "        ")
        assert aligned == (
            "        model = XGB()\n"
            "        if use_gpu:\n"
            "            model = XGB(tree_method='gpu_hist')\n"
            "        model.fit(X, y)"
        )

    def test_skips_blank_lines(self) -> None:
        aligned = align_replacement_indent("model = XGB()\n\nmodel.fit(X, y)", "    ")
        assert aligned == "    model = XGB()\n\n    model.fit(X, y)"

    def test_empty_replacement_stays_empty(self) -> None:
        assert align_replacement_indent("", "    ") == ""


class TestModelCard:
    """Test the `ModelCard` structured-output model."""

    def test_instantiates(self) -> None:
        card = ModelCard(
            model_name="CatBoostClassifier",
            rationale="Handles categorical features natively.",
            example_code="from catboost import CatBoostClassifier",
            library_dependencies=["catboost"],
        )
        assert card.model_name == "CatBoostClassifier"
        assert card.library_dependencies == ["catboost"]

    def test_strips_markdown_fences_from_example_code(self) -> None:
        card = ModelCard(
            model_name="CatBoost",
            rationale="good",
            example_code="```python\nmodel = CatBoostClassifier()\n```",
            library_dependencies=["catboost"],
        )
        assert card.example_code == "model = CatBoostClassifier()"
        assert "```" not in card.example_code

    def test_preserves_raw_example_code(self) -> None:
        raw = "import xgboost as xgb\nmodel = xgb.XGBRegressor()"
        card = ModelCard(
            model_name="XGBoost",
            rationale="strong",
            example_code=raw,
            library_dependencies=["xgboost"],
        )
        assert card.example_code == raw

    def test_preserves_non_parsing_raw_fragment(self) -> None:
        fragment = "if x > 1:  # incomplete standalone fragment"
        card = ModelCard(
            model_name="Frag",
            rationale="r",
            example_code=fragment,
            library_dependencies=[],
        )
        assert card.example_code == fragment

    def test_requires_non_empty_model_name(self) -> None:
        with pytest.raises(ValidationError):
            ModelCard(
                model_name="",
                rationale="r",
                example_code="x = 1",
                library_dependencies=[],
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelCard(
                model_name="M",
                rationale="r",
                example_code="x = 1",
                library_dependencies=[],
                surprise="nope",
            )

    def test_json_round_trip(self) -> None:
        card = ModelCard(
            model_name="LightGBM",
            rationale="fast",
            example_code="import lightgbm",
            library_dependencies=["lightgbm", "numpy"],
        )
        restored = ModelCard.model_validate_json(card.model_dump_json())
        assert restored == card


class TestRetrievedCandidates:
    """Test the `RetrievedCandidates` container."""

    def test_instantiates(self) -> None:
        candidates = [
            ModelCard(
                model_name="A",
                rationale="r",
                example_code="x = 1",
                library_dependencies=[],
            )
        ]
        container = RetrievedCandidates(
            candidates=candidates,
            query_used="CTR prediction SOTA",
            total_found=1,
        )
        assert container.total_found == 1
        assert container.candidates[0].model_name == "A"

    def test_json_round_trip_nests_cards(self) -> None:
        card = ModelCard(
            model_name="B",
            rationale="r",
            example_code="y = 2",
            library_dependencies=["sklearn"],
        )
        container = RetrievedCandidates(
            candidates=[card],
            query_used="q",
            total_found=1,
        )
        restored = RetrievedCandidates.model_validate_json(container.model_dump_json())
        assert restored == container
        assert restored.candidates[0].library_dependencies == ["sklearn"]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RetrievedCandidates(
                candidates=[],
                query_used="q",
                total_found=0,
                extra=True,
            )


class TestAblationVariant:
    """Test the `AblationVariant` ablation experiment descriptor."""

    def test_instantiates(self) -> None:
        variant = AblationVariant(
            variant_id="v1",
            component_name="imputer",
            category=ComponentCategory.DATA_PREPROCESSING,
            hypothesis="Median imputation beats mean.",
            modified_code_block="('imputer', SimpleImputer(strategy='median'))",
            ablation_code="import pandas as pd\n...",
        )
        assert variant.variant_id == "v1"
        assert variant.category is ComponentCategory.DATA_PREPROCESSING

    def test_rejects_unknown_category(self) -> None:
        with pytest.raises(ValidationError):
            AblationVariant(
                variant_id="v1",
                component_name="imputer",
                category="TELEPATHY",  # type: ignore[arg-type]
                hypothesis="h",
                modified_code_block="b",
                ablation_code="c",
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AblationVariant(
                variant_id="v1",
                component_name="imputer",
                category=ComponentCategory.DATA_PREPROCESSING,
                hypothesis="h",
                modified_code_block="b",
                ablation_code="c",
                nonsense=1,
            )


class TestAblationReport:
    """Test the `AblationReport` aggregation."""

    def _report(self) -> AblationReport:
        return AblationReport(
            baseline_score=0.90,
            ablation_results=[
                AblationResultItem(
                    variant_id="v1",
                    validation_score=0.95,
                    delta_from_baseline=0.05,
                    summary="median imputation helped",
                ),
                AblationResultItem(
                    variant_id="v2",
                    validation_score=0.88,
                    delta_from_baseline=-0.02,
                    summary="removed feature hurt",
                ),
                AblationResultItem(
                    variant_id="v3",
                    validation_score=0.93,
                    delta_from_baseline=0.03,
                    summary="scaling helped slightly",
                ),
            ],
            highest_impact_component="imputer",
            raw_log_summary="...",
        )
        pass

    def test_instantiates(self) -> None:
        report = self._report()
        assert report.baseline_score == pytest.approx(0.90)
        assert len(report.ablation_results) == 3
        assert report.highest_impact_component == "imputer"

    def test_highest_impact_result_returns_max_delta(self) -> None:
        report = self._report()
        top = report.highest_impact_result()
        assert top is not None
        assert top.variant_id == "v1"
        assert top.delta_from_baseline == pytest.approx(0.05)

    def test_highest_impact_result_none_when_empty(self) -> None:
        report = AblationReport(
            baseline_score=0.5,
            ablation_results=[],
            highest_impact_component="",
            raw_log_summary="",
        )
        assert report.highest_impact_result() is None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AblationReport(
                baseline_score=0.5,
                ablation_results=[],
                highest_impact_component="",
                raw_log_summary="",
                ghost="x",
            )


class TestTargetCodeBlock:
    """Test `TargetCodeBlock` extraction and AST-safe replacement."""

    def _block(self, **overrides: object) -> TargetCodeBlock:
        defaults: dict[str, object] = {
            "raw_code": "x = 1\ny = 2",
            "category": ComponentCategory.FEATURE_ENGINEERING,
            "start_line": None,
            "end_line": None,
            "initial_plan": "scale the features",
        }
        defaults.update(overrides)
        return TargetCodeBlock(**defaults)

    def test_instantiates(self) -> None:
        block = self._block()
        assert block.category is ComponentCategory.FEATURE_ENGINEERING
        assert block.start_line is None
        assert block.end_line is None
        assert block.initial_plan == "scale the features"

    def test_replace_in_replaces_substring(self) -> None:
        full = "def pipeline():\n    x = 1\n    y = 2\n    return x + y\n"
        block = self._block()
        new_code = "x = 10\ny = 20"
        result = block.replace_in(full, new_code)
        assert result == "def pipeline():\n    x = 10\n    y = 20\n    return x + y\n"

    def test_replace_in_uses_line_numbers_when_given(self) -> None:
        full = "a = 0\nx = 1\ny = 2\nb = 3\n"
        block = self._block(start_line=2, end_line=3)
        result = block.replace_in(full, "x = 100\ny = 200")
        assert result == "a = 0\nx = 100\ny = 200\nb = 3\n"

    def test_replace_in_raises_when_block_not_found(self) -> None:
        full = "def pipeline():\n    z = 9\n    return z\n"
        block = self._block()
        with pytest.raises(ValueError, match="not found"):
            block.replace_in(full, "z = 1")

    def test_replace_in_raises_on_invalid_result_syntax(self) -> None:
        full = "def f():\n    x = 1\n    return x\n"
        block = self._block(raw_code="x = 1")
        with pytest.raises(ValueError, match="invalid Python"):
            block.replace_in(full, "if True:")

    def test_replace_in_raises_on_empty_raw_code(self) -> None:
        full = "def f():\n    x = 1\n    return x\n"
        block = self._block(raw_code="   \n  ")
        with pytest.raises(ValueError, match="empty"):
            block.replace_in(full, "x = 2")

    def test_replace_in_class_method_with_unindented_replacement(self) -> None:
        full = (
            "class Trainer:\n"
            "    def fit(self):\n"
            "        model = LogisticRegression()\n"
            "        model.fit(X, y)\n"
            "        return model\n"
        )
        block = self._block(raw_code="model = LogisticRegression()\nmodel.fit(X, y)")
        result = block.replace_in(full, "model = XGBClassifier()\nmodel.fit(X, y)")
        assert result == (
            "class Trainer:\n"
            "    def fit(self):\n"
            "        model = XGBClassifier()\n"
            "        model.fit(X, y)\n"
            "        return model\n"
        )

    def test_replace_in_class_method_with_preindented_replacement(self) -> None:
        full = (
            "class Trainer:\n"
            "    def fit(self):\n"
            "        model = LogisticRegression()\n"
            "        model.fit(X, y)\n"
            "        return model\n"
        )
        block = self._block(raw_code="model = LogisticRegression()\nmodel.fit(X, y)")
        result = block.replace_in(
            full, "        model = XGBClassifier()\n        model.fit(X, y)"
        )
        assert result == (
            "class Trainer:\n"
            "    def fit(self):\n"
            "        model = XGBClassifier()\n"
            "        model.fit(X, y)\n"
            "        return model\n"
        )

    def test_json_round_trip(self) -> None:
        block = self._block(start_line=1, end_line=2)
        restored = TargetCodeBlock.model_validate_json(block.model_dump_json())
        assert restored == block

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            self._block(extra="x")


class TestRefinementPlan:
    """Test the `RefinementPlan` inner-loop planning model."""

    def test_instantiates(self) -> None:
        plan = RefinementPlan(
            plan_id="p3",
            natural_language_plan="Switch to a ranker with pairwise loss.",
            target_subcomponents=["loss", "model_head"],
            expected_gain="+0.02 NDCG@10",
            iteration_index=3,
        )
        assert plan.plan_id == "p3"
        assert plan.iteration_index == 3
        assert plan.target_subcomponents == ["loss", "model_head"]

    def test_json_round_trip(self) -> None:
        plan = RefinementPlan(
            plan_id="p1",
            natural_language_plan="try median imputation",
            target_subcomponents=["imputer"],
            expected_gain="+0.01",
            iteration_index=1,
        )
        restored = RefinementPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RefinementPlan(
                plan_id="p1",
                natural_language_plan="plan",
                target_subcomponents=[],
                expected_gain="+0",
                iteration_index=1,
                stray=1,
            )
