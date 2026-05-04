from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

from ragpill.eval_types import (
    EvaluationReason,
    EvaluatorContext,
    _build_serialization_arguments,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from ragpill.settings import MLFlowSettings


def default_input_to_key(input: Any) -> str:
    """Convert a task input to a deterministic string key.

    MD5 hash of the stringified input. Used internally by
    :mod:`ragpill.execution` to namespace per-case trace data.

    Args:
        input: The task input value (any type).

    Returns:
        A hex-encoded MD5 digest of the stringified input.
    """
    return hashlib.md5(str(input).encode()).hexdigest()


class TestCaseMetadata(BaseModel):
    """
    In general: For non-global evaluators the evaluator metadata takes precedence over case metadata.
    For global evaluators, the case metadata takes precedence over evaluator metadata.
    This is to allow global evaluators to set default expected values, which can be
    overridden by case metadata.
    """

    expected: bool | None = Field(
        default=None,
        description="Expected evaluation result. Defaults to True. Set to False, if you want to assure a certain fact is NOT in the answer. Useful to check for hallucinations.",
    )
    attributes: dict[str, Any] = Field(default_factory=dict)
    tags: set[str] = Field(default_factory=set)
    repeat: int | None = Field(
        default=None,
        ge=1,
        description="Per-case override: number of times to run this test case. None defers to MLFlowSettings.ragpill_repeat.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-case override: minimum fraction of runs that must pass. None defers to MLFlowSettings.ragpill_threshold.",
    )


CaseMetadataT = TypeVar("CaseMetadataT", bound=TestCaseMetadata)


class EvaluatorMetadata(BaseModel):
    """Metadata for LLM evaluation evaluators."""

    expected: bool | None = Field(
        default=None,
        description="Expected evaluation result. Defaults to True. Set to False, if you want to assure a certain fact is NOT in the answer. Useful to check for hallucinations.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes for the evaluator. e.g. 'importance':'high' or 'category':'factual'. Will be merged with Case metadata attributes during evaluation (precedence: Case Evaluator attribute > Case attribute > Global Evaluator attribute). unique key will be a column in the final Test Report.",
    )
    tags: set[str] = Field(
        default_factory=set,
        description="Tags for categorizing the evaluator. Can be used to filter evaluators later on.",
    )
    is_global_evaluator: bool = Field(
        default=False,
        description="Indicates whether this evaluator is a global evaluator (applies to all cases) or specific to a case.",
    )
    other_evaluator_data: str = Field(
        default="",
        description="Other relevant data for the evaluator, like rubric of LLMJudge, source list of SourceInListEvaluator, etc.",
    )


def merge_metadata(
    case_metadata: TestCaseMetadata,
    evaluator_metadata: EvaluatorMetadata,
) -> EvaluatorMetadata:
    """Merge case and evaluator metadata into a single resolved metadata.

    Precedence depends on whether the evaluator is global:

    - **Non-global evaluators:** evaluator attribute > case attribute.
    - **Global evaluators:** case attribute > evaluator attribute.

    The ``expected`` flag defaults to ``True`` when neither source sets it.
    Tags are always unioned from both sources.

    Args:
        case_metadata: Metadata attached to the test case.
        evaluator_metadata: Metadata attached to the evaluator instance.

    Returns:
        A new ``EvaluatorMetadata`` with merged attributes, tags, and resolved
        ``expected`` flag.
    """
    merged_metadata = evaluator_metadata.model_copy()
    if merged_metadata.is_global_evaluator:
        merged_metadata.attributes |= case_metadata.attributes
        merged_metadata.expected = (
            case_metadata.expected if case_metadata.expected is not None else merged_metadata.expected
        )
    else:
        merged_metadata.attributes = case_metadata.attributes | merged_metadata.attributes
        merged_metadata.expected = (
            evaluator_metadata.expected if evaluator_metadata.expected is not None else case_metadata.expected
        )

    merged_metadata.expected = merged_metadata.expected if merged_metadata.expected is not None else True

    merged_metadata.tags = merged_metadata.tags | case_metadata.tags

    return merged_metadata


def dict_factory(x: list[tuple[str, Any]]) -> dict[str, Any]:
    exclude_fields = ("id", "expected", "attributes", "tags", "is_global", "model")
    return {k: v for (k, v) in x if ((v is not None) and (k not in exclude_fields))}


@dataclass
class BaseEvaluator:
    """Base class for all ragpill evaluators.

    All custom evaluators must inherit from this class and implement:

    1. [`from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line] class method -
       for CSV integration with [`load_testset`][ragpill.csv.testset.load_testset]
    2. `run` async method - for evaluation logic

    Attributes:
        evaluation_name: Unique identifier for this evaluator instance

        expected: Whether we expect this check to pass. Defaults to None, which means
            the value is inherited from the case's TestCaseMetadata.expected at evaluation
            time. If neither evaluator nor case metadata sets it, defaults to True.
            For non-global evaluators, an explicit evaluator value takes precedence over
            case metadata. For global evaluators, case metadata takes precedence.        attributes: Dictionary for additional metadata (populated from extra CSV columns)
        tags: List of tags for organization and filtering
        is_global: Whether this evaluator applies to all test cases

    Note:
        The 'check' parameter is only used in from_csv_line() to pass configuration
        when creating the evaluator - it's not stored as a class attribute.

    See Also:
        [`ragpill.csv.testset.load_testset`][ragpill.csv.testset.load_testset]:
            Create datasets from CSV files
    """

    evaluation_name: uuid.UUID = field(default_factory=uuid.uuid4)
    expected: bool | None = field(default=None)
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    is_global: bool = field(default=False)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the class name used to identify this evaluator.

        Returns:
            The evaluator's class name.
        """
        return cls.__name__

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Return a dict of non-default field values for this evaluator.

        Iterates over the dataclass fields and returns those whose value differs
        from the declared default (either ``field.default`` or
        ``field.default_factory()``). Useful for logging/debugging evaluator
        configuration.

        Returns:
            A dictionary mapping field names to their non-default values.
        """
        return _build_serialization_arguments(self)

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> BaseEvaluator:
        """Create an evaluator from a CSV line.

        This class method is required for CSV integration with
        [`load_testset`][ragpill.csv.testset.load_testset].
        The signature must be exactly as shown. Subclasses can override this method to
        customize how they parse the check parameter or handle additional configuration.

        Custom Attributes:
            Any additional CSV columns beyond the standard ones (Question, test_type, expected,
            tags, check) will be passed as **kwargs and stored in the evaluator's
            attributes dict. These can be used for metadata tracking, filtering, or custom logic.

            If all evaluators for a question share the same attribute value, that attribute
            becomes part of the Test Case metadata and will be visible in MLflow.

        Parameterization Patterns:
            There are two ways to parameterize custom evaluators:

            1. **Environment Variables** (for shared config across all instances):
               Use pydantic-settings BaseSettings to load from environment variables.
               Good for API keys, global thresholds, model names, etc.

            2. **JSON in check column** (for per-instance config):
               Parse JSON from the check parameter to get per-test configuration.
               Good for regex patterns, specific values, test-specific thresholds.

        Args:
            expected: Whether we expect this check to pass.
                     Set to `true` for normal tests (e.g., "answer should mention Paris").
                     Set to `false` for negative tests (e.g., "answer should NOT hallucinate links").
                     The evaluation result is compared against this expectation.
                     When constructing evaluators programmatically (not via CSV), you can
                     omit this to inherit the value from case metadata at evaluation time.
            tags: Comma-separated tags string from CSV for categorization and filtering.
            check: Evaluator-specific configuration data. Can be JSON string or plain text.
                   For JSON: Will be parsed and passed as **check_params to the evaluator.
                   For plain text: Subclasses should override this method to handle their format.
            **kwargs: Additional attributes from extra CSV columns (e.g., priority, category).
                     These become part of `evaluator.attributes` and are used for:
                     - Metadata tracking and filtering
                     - MLflow logging (when shared across all evaluators of a question)
                     - Custom evaluation logic in your evaluators

        Returns:
            Instance of the evaluator class

        Raises:
            NotImplementedError: If check is not valid JSON and subclass hasn't overridden this method

        Example:
            For CSV usage examples, see the
            [CSV Adapter Guide](https://joelgotsch.github.io/ragpill/latest/guide/csv-adapter/) and
            [Custom Evaluators Guide](https://joelgotsch.github.io/ragpill/latest/guide/evaluators/).

            ```python
            class MyEvaluator(BaseEvaluator):
                pattern: str

                @classmethod
                def from_csv_line(cls, expected: bool, tags: set[str],
                                check: str, **kwargs):
                    # Parse check parameter (JSON or plain text)
                    try:
                        check_dict = json.loads(check)
                        pattern = check_dict.get('pattern', check)
                    except json.JSONDecodeError:
                        pattern = check  # Use as-is

                    return cls(
                        expected=expected,
                        tags=tags,
                        attributes=kwargs,  # Contains custom CSV columns
                        pattern=pattern,
                    )
            ```
        """

        # Try to parse check as JSON, if it fails treat as plain text
        check_params: dict[str, Any] = {}
        if check:
            try:
                check_params = json.loads(check)
            except json.JSONDecodeError:
                # If not JSON, subclasses should override this method
                # to handle their specific format
                raise NotImplementedError(
                    f"Subclasses must implement from_csv_line to handle non-JSON check format: {check}"
                )

        return cls(expected=expected, tags=tags, attributes=kwargs, **check_params)

    @property
    def metadata(self) -> EvaluatorMetadata:
        """Build metadata from evaluator fields."""
        return EvaluatorMetadata(
            expected=self.expected,
            attributes=self.attributes,
            tags=self.tags,
            is_global_evaluator=self.is_global,
            other_evaluator_data=str(asdict(self, dict_factory=dict_factory)),
        )

    async def run(
        self,
        ctx: EvaluatorContext[Any, Any, EvaluatorMetadata],  # pyright: ignore[reportUnusedParameter]  # ctx used by subclasses
    ) -> EvaluationReason:
        """Implement the evaluation logic. Overwrite this in subclasses.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result with a boolean value and reason string.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    async def evaluate(
        self,
        ctx: EvaluatorContext[Any, Any, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Run the evaluator and apply the ``expected`` polarity logic.

        Calls :meth:`run` and then flips the result value when the merged
        metadata's ``expected`` flag is ``False``.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result with the ``expected`` polarity applied.
        """
        # handle common logic for expected:
        eval_result = await self.run(ctx)

        # if eval_result.value is None:
        #     return eval_result
        assert isinstance(eval_result.value, bool), "Evaluator must return a boolean value."
        assert isinstance(ctx.metadata, TestCaseMetadata), "Expected TestCaseMetadata from context."
        merged_metadata = merge_metadata(case_metadata=ctx.metadata, evaluator_metadata=self.metadata)
        eval_result.value = eval_result.value == merged_metadata.expected
        return eval_result


def resolve_repeat(case_metadata: TestCaseMetadata | None, settings: MLFlowSettings) -> tuple[int, float]:
    """Resolve effective repeat count and pass threshold from per-case override or global default.

    Args:
        case_metadata: Per-case metadata (may be None or have None fields).
        settings: Global MLFlowSettings providing default repeat/threshold.

    Returns:
        Tuple of (repeat, threshold) with per-case values taking precedence over globals.
    """
    repeat = case_metadata.repeat if (case_metadata and case_metadata.repeat is not None) else settings.ragpill_repeat
    threshold = (
        case_metadata.threshold
        if (case_metadata and case_metadata.threshold is not None)
        else settings.ragpill_threshold
    )
    return repeat, threshold
