"""Unified execution guardrail pipeline (leakage -> usage -> sandbox -> debugger).

Routes every script execution through a single reusable orchestrator: the
data leakage guardrail, the data usage guardrail, the subprocess sandbox,
and the automatic debugger loop, returning a validated ``ExecutionResult``.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.settings import ModelSettings

from problem_2_v2.contracts.task import ExecutionResult, TaskSpecification
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent, LeakageEnforcementError
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner


class ExecutionConfig(BaseModel):
    """Configuration for the execution guardrail pipeline.

    Attributes:
        timeout_seconds: Per-script sandbox wall-clock timeout.
        max_debug_rounds: Debugger repair budget.
        sandbox_base_dir: Root directory holding per-run sandboxes.
        enable_leakage_check: Whether to run the data leakage guardrail.
        enable_usage_check: Whether to run the data usage guardrail.
        production_timeout_seconds: Extended timeout for full-dataset
            finalization runs.
        max_leakage_retries: Maximum check→repair→re-check cycles when
            leakage is detected. ``0`` means a single audit attempt with
            no retries.
        strict_leakage: When ``True``, raises ``LeakageEnforcementError``
            if leakage persists after all retries. When ``False``
            (default), warns and continues.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timeout_seconds: int = Field(default=1200, description="Sandbox timeout in seconds.")
    max_debug_rounds: int = Field(default=3, description="Debugger repair budget.")
    sandbox_base_dir: str = Field(default="runs", description="Sandbox root directory.")
    enable_leakage_check: bool = Field(default=True, description="Run the leakage guardrail.")
    enable_usage_check: bool = Field(default=True, description="Run the usage guardrail.")
    production_timeout_seconds: int = Field(
        default=3600,
        description="Extended timeout for full-dataset finalization.",
    )
    max_leakage_retries: int = Field(
        default=5,
        description="Max check→repair→re-check cycles for leakage.",
    )
    strict_leakage: bool = Field(
        default=False,
        description="Raise LeakageEnforcementError if leakage persists after retries.",
    )


class ExecutionGuardrailPipeline:
    """Orchestrates guardrails, sandbox execution, and the debugger loop.

    The ``last_*`` attributes record the most recent synchronous run and
    are read immediately after ``run()``; callers must not invoke ``run``
    concurrently on the same instance.

    Attributes:
        config: Pipeline configuration.
        leakage: Data leakage checker agent.
        usage: Data usage checker agent.
        runner: Sandbox subprocess runner.
        debugger: Automatic debugger agent.
        last_guarded_code: The most recently guarded (post-check) code.
        last_executed_code: The most recently executed code (post-debug).
        last_debug_rounds: Repair rounds used by the most recent run.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        leakage: DataLeakageCheckerAgent | None = None,
        usage: DataUsageCheckerAgent | None = None,
        runner: SubprocessRunner | None = None,
        debugger: DebuggerAgent | None = None,
        model: str = "openai:gpt-4o",
        model_settings: ModelSettings | dict | None = None,
    ) -> None:
        """Create the execution guardrail pipeline.

        Args:
            config: Pipeline configuration (defaults to ``ExecutionConfig``).
            leakage: Data leakage checker agent.
            usage: Data usage checker agent.
            runner: Sandbox subprocess runner.
            debugger: Automatic debugger agent.
            model: Pydantic AI model string used for any default agents.
            model_settings: Optional LLM generation settings (e.g. max_tokens).
        """
        self.config = config or ExecutionConfig()
        self.model_settings = model_settings
        self.leakage = leakage or DataLeakageCheckerAgent(model=model, model_settings=model_settings)
        self.usage = usage or DataUsageCheckerAgent(model=model, model_settings=model_settings)
        self.runner = runner or SubprocessRunner(
            runs_dir=self.config.sandbox_base_dir,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.debugger = debugger or DebuggerAgent(
            runner=self.runner,
            model=model,
            max_debug_rounds=self.config.max_debug_rounds,
            model_settings=model_settings,
        )
        self.last_guarded_code: str | None = None
        self.last_executed_code: str | None = None
        self.last_debug_rounds: int = 0

    def guard(self, code: str, spec: TaskSpecification) -> str:
        """Apply the leakage and usage passes to a script.

        The leakage pass runs a retry loop of up to
        ``config.max_leakage_retries`` check→repair→re-check cycles.
        When ``config.strict_leakage`` is ``True`` and leakage persists
        after all retries, ``LeakageEnforcementError`` is raised.

        Guardrail passes degrade gracefully: when an LLM check fails the
        pass is skipped with a warning and the code is left unchanged.

        Args:
            code: The solution script to audit.
            spec: The task specification (dataset metadata source).

        Returns:
            The guarded script, possibly repaired or augmented by the
            active guardrail checks.

        Raises:
            LeakageEnforcementError: When ``strict_leakage`` is enabled
                and leakage persists after all retry attempts.
        """
        guarded = code
        if self.config.enable_leakage_check:
            with logfire.span("execution.leakage_check"):
                guarded = self._leakage_guard_loop(guarded)
        if self.config.enable_usage_check:
            with logfire.span("execution.usage_check"):
                try:
                    usage_status = self.usage.audit(spec, guarded)
                    if usage_status.improved_code_block:
                        guarded = usage_status.improved_code_block
                except Exception as exc:
                    logfire.warn("execution.usage_check.failed", error=str(exc))
        self.last_guarded_code = guarded
        return guarded

    def _leakage_guard_loop(self, code: str) -> str:
        """Run the check→repair→re-check cycle with retry budget.

        Args:
            code: The solution script to audit.

        Returns:
            The best-effort repaired code.

        Raises:
            LeakageEnforcementError: When ``strict_leakage`` is enabled
                and leakage persists after all retry attempts.
        """
        guarded = code
        retries_used = 0
        leaking = False

        try:
            for attempt in range(1 + self.config.max_leakage_retries):
                status, guarded = self.leakage.audit(guarded)
                if not status.is_leaking:
                    if attempt > 0:
                        # Leakage was detected on a prior attempt and is
                        # now resolved after repair.
                        logfire.info(
                            "execution.leakage_repaired",
                            retries_used=attempt,
                            strict=self.config.strict_leakage,
                        )
                    leaking = False
                    break
                leaking = True
                retries_used = attempt + 1
        except Exception as exc:
            logfire.warn("execution.leakage_check.failed", error=str(exc))
            return guarded

        if leaking:
            logfire.warn(
                "execution.leakage_unrepaired",
                retries_used=retries_used,
                strict=self.config.strict_leakage,
            )
            if self.config.strict_leakage:
                raise LeakageEnforcementError(
                    f"Data leakage persists after {retries_used} "
                    f"repair attempt(s). Aborting execution."
                )
        return guarded

    def run(
        self,
        code: str,
        spec: TaskSpecification,
        run_id: str = "exec",
        candidate_id: str = "candidate",
    ) -> ExecutionResult:
        """Execute code through guardrails, sandbox, and the debugger loop.

        Args:
            code: The candidate Python script to execute.
            spec: The task specification.
            run_id: Identifier of the current run.
            candidate_id: Identifier of the candidate script.

        Returns:
            A validated ``ExecutionResult`` with success flag, stdout,
            stderr, parsed validation score, and runtime duration.
        """
        with logfire.span("execution.run", run_id=run_id, candidate_id=candidate_id):
            guarded = self.guard(code, spec)
            with logfire.span("execution.sandbox_exec"):
                outcome = self.debugger.debug(
                    guarded,
                    run_id=run_id,
                    candidate_id=candidate_id,
                    dataset_dir=spec.dataset_dir,
                    dataset_files=spec.dataset_files,
                )
            self.last_executed_code = outcome.code
            self.last_debug_rounds = outcome.debug_rounds
            return outcome.result
