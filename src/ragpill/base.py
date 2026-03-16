import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, Field
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluationReason, Evaluator, InputsT, OutputT


def default_input_to_key(input: InputsT) -> str:
    """Default function to convert input to a string key."""
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


CaseMetadataT = TypeVar("MetadataT", bound=TestCaseMetadata)


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


EvaluatorMetadataT = TypeVar("EvaluatorMetadataT", bound=EvaluatorMetadata)


def merge_metadata(
    case_metadata: TestCaseMetadata,
    evaluator_metadata: EvaluatorMetadata,
) -> EvaluatorMetadata:
    """Merge case and evaluator metadata, with precedence: Case Evaluator attribute > Case attribute > Global Evaluator attribute."""
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


def dict_factory(x):
    exclude_fields = ("id", "expected", "attributes", "tags", "_is_global", "model")
    return {k: v for (k, v) in x if ((v is not None) and (k not in exclude_fields))}


@dataclass
class BaseEvaluator(Evaluator):
    """Base class for all evaluators.

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
        _is_global: Whether this evaluator applies to all test cases

    Note:
        The 'check' parameter is only used in from_csv_line() to pass configuration
        when creating the evaluator - it's not stored as a class attribute.

    See Also:
        [`ragpill.csv.testset.load_testset`][ragpill.csv.testset.load_testset]:
            Create datasets from CSV files
    """

    evaluation_name: uuid.UUID = field(
        default_factory=uuid.uuid4
    )  # this is used by pydantic-ai to create the name of the reportcase.assertion
    expected: bool | None = field(default=None)
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    _is_global: bool = field(default=False)

    @classmethod
    def from_csv_line(
        cls, expected: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "BaseEvaluator":
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
            [CSV Adapter Guide](https://your-docs-url.com/guide/csv-adapter/) and
            [Custom Evaluators Tutorial](https://your-docs-url.com/tutorials/custom-evaluators/).

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
        check_params = {}
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
            is_global_evaluator=self._is_global,
            other_evaluator_data=str(asdict(self, dict_factory=dict_factory)),
        )

    async def run(
        self,
        ctx: EvaluatorContext[InputsT, OutputT, CaseMetadataT],
    ) -> EvaluationReason:
        """
        The method to implement the evaluation logic. Overwrite this in subclasses.

        :param ctx: Description
        :type ctx: EvaluatorContext[InputsT, OutputT, CaseMetadataT]
        :return: Description
        :rtype: EvaluationReason
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    async def evaluate(
        self,
        ctx: EvaluatorContext[InputsT, OutputT, EvaluatorMetadataT],
    ) -> EvaluationReason:
        # handle common logic for expected:
        eval_result = await self.run(ctx)

        # if eval_result.value is None:
        #     return eval_result
        assert isinstance(eval_result.value, bool), "Evaluator must return a boolean value."
        merged_metadata = merge_metadata(case_metadata=ctx.metadata, evaluator_metadata=self.metadata)
        eval_result.value = eval_result.value == merged_metadata.expected
        return eval_result
