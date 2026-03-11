import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities import Document, SpanType, Trace
from pydantic_ai import models
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.context import InputsT, OutputT
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output, judge_output

from ragpill.base import BaseEvaluator, EvaluatorMetadata, default_input_to_key
from ragpill.settings import LLMJudgeSettings, MLFlowSettings
from ragpill.utils import _extract_markdown_quotes, _normalize_text, get_pydantic_ai_llm_model


def _get_default_judge_llm() -> models.Model:
    """Get default LLMJudge settings instance."""
    settings = LLMJudgeSettings()
    if not settings.api_key or not settings.base_url or not settings.model_name:
        raise ValueError(
            "LLMJudgeSettings must have api_key, base_url, and model_name set to get default LLM model. Set them via environment variables RAGPILL_LLMJUDGE_API_KEY, RAGPILL_LLMJUDGE_BASE_URL, and RAGPILL_LLMJUDGE_MODEL_NAME respectively."
        )
    return get_pydantic_ai_llm_model(
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        model_name=settings.model_name,
        temperature=settings.temperature,
    )


@dataclass(kw_only=True, repr=False)
class LLMJudge(BaseEvaluator):
    """The LLMJudge evaluator uses a language model to judge whether an output meets specified rubric.

    A rubric usually is one of the following:
    - A fact that the output should contain or not contain (rubric="Output must contain the fact that Paris is the capital of France.")
    - About the style of the output (rubric="Output should be in a formal tone." or "Output should be in German")

    **Note**:
    Avoid complex instructions in the rubric, as the model may not follow them reliably. Instead, try to break it down into multiple instances of the LLMJudge.

    """

    rubric: str
    model: models.Model = field(repr=False, default_factory=_get_default_judge_llm)
    include_input: bool = field(default=False)

    @classmethod
    def from_csv_line(
        cls,
        expected: bool,
        mandatory: bool,
        tags: set[str],
        check: str,
        get_llm: Callable[[], models.Model] = _get_default_judge_llm,
        **kwargs: Any,
    ) -> "LLMJudge":
        """Create an LLMJudge from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        For LLMJudge, the check parameter is treated as the rubric text.
        If check is a JSON object with a 'rubric' key, that value is used.
        Otherwise, the entire check string is used as the rubric.

        Args:
            expected: Expected evaluation result
            mandatory: Whether evaluation is mandatory
            tags: Comma-separated tags string
            check: Rubric text or JSON with 'rubric' key
            get_llm: Callable that returns a Model instance (defaults to get_default_judge_llm)
            **kwargs: Additional parameters (can include 'model' to override default)

        Note: The model parameter must be provided. It should come from:
        - Dependency injection (e.g., a module-level or class-level settings object)
        - The check column as JSON: {"rubric": "...", "model": "openai:gpt-4o"}
        - An additional CSV column named 'model'
        """

        if not check:
            raise ValueError("LLMJudge requires a non-empty 'check' parameter for the rubric.")
        try:
            check_obj = json.loads(check)
            if isinstance(check_obj, dict):
                rubric = check_obj.pop("rubric", check)
                kwargs.update(check_obj)
        except json.JSONDecodeError:
            # Plain text - use as rubric
            rubric = check
        model = kwargs.pop("model", None) or get_llm()

        return cls(
            rubric=rubric,
            model=model,
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        if self.include_input:
            grading_output = await judge_input_output(ctx.inputs, ctx.output, self.rubric, self.model)
        else:
            grading_output = await judge_output(ctx.output, self.rubric, self.model)
        return EvaluationReason(
            value=grading_output.pass_,
            reason=grading_output.reason,
        )

    def build_serialization_arguments(self) -> dict[str, Any]:
        result = super(BaseEvaluator, self).build_serialization_arguments()
        # always serialize the model as a string when present; use its name if it's a KnownModelName
        if (model := result.get("model")) and isinstance(model, models.Model):  # pragma: no branch
            result["model"] = f"{model.system}:{model.model_name}"
        return result

    @property
    def metadata(self) -> EvaluatorMetadata:
        """Build metadata from evaluator fields.

        the default in BaseEvaluator is overridden because excluding the not pickleable model field seems impossible
        """
        return EvaluatorMetadata(
            expected=self.expected,
            mandatory=self.mandatory,
            attributes=self.attributes,
            tags=self.tags,
            is_global_evaluator=self._is_global,
            other_evaluator_data=self.rubric,
        )


@dataclass(kw_only=True, repr=False)
class WrappedPydanticEvaluator(BaseEvaluator):
    """Wrapper to use any pydantic-evals Evaluator as a ragpill BaseEvaluator.
    See https://ai.pydantic.dev/evals/evaluators/overview/ for a list.
    Limitation: Span-Based evaluators are not supported as logfire is not supported in ragpill yet.

    **Note**:
    If you want to use pydantic-evals evaluators in your csv-defined testsets,
    you need to define a subclass of this class that implements from_csv_line to
    create the specific pydantic evaluator.

    Attributes:
        pydantic_evaluator: The pydantic-evals Evaluator instance to wrap.

    Example:
        ```python
        from pydantic_evals.evaluators import SomePydanticEvaluator
        from ragpill.base import WrappedPydanticEvaluator
        ragpill_evaluator = WrappedPydanticEvaluator(
            pydantic_evaluator=SomePydanticEvaluator(...),
            expected=True,
            mandatory=False,
            tags={"tag1", "tag2"},
            attributes={"attr1": "value1"},
        )
    """

    pydantic_evaluator: Evaluator = field(repr=False)

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        eval_output = await self.pydantic_evaluator.evaluate(ctx)
        assert isinstance(eval_output, EvaluationReason), (
            "Wrapped pydantic evaluator did not return an EvaluationReason."
        )
        return eval_output


@dataclass(kw_only=True, repr=False)
class SpanBaseEvaluator(BaseEvaluator):
    """
    This base class that retrieves the spans from mlflow trace.
    This allows subclasses to implement evaluation logic based on spans.
    Why is this useful? See https://ai.pydantic.dev/evals/evaluators/span-based/

    **Why Span-Based Evaluation?**
    Traditional evaluators assess task inputs and outputs. For simple tasks, this may be sufficient—if the output is correct, the task succeeded. But for complex multi-step agents, the process matters as much as the result:

    **A correct answer reached incorrectly** - An agent might produce the right output by accident (e.g., guessing, using cached data when it should have searched, calling the wrong tools but getting lucky)
    **Verification of required behaviors** - You need to ensure specific tools were called, certain code paths executed, or particular patterns followed
    **Performance and efficiency** - The agent should reach the answer efficiently, without unnecessary tool calls, infinite loops, or excessive retries
    **Safety and compliance** - Critical to verify that dangerous operations weren't attempted, sensitive data wasn't accessed inappropriately, or guardrails weren't bypassed

    **Real-World Scenarios**
    Span-based evaluation is particularly valuable for:

    **RAG systems** - Verify documents were retrieved and reranked before generation, not just that the answer included citations
    **Multi-agent coordination** - Ensure the orchestrator delegated to the right specialist agents in the correct order
    **Tool-calling agents** - Confirm specific tools were used (or avoided), and in the expected sequence
    **Debugging and regression testing** - Catch behavioral regressions where outputs remain correct but the internal logic deteriorates
    **Production alignment** - Ensure your evaluation assertions operate on the same telemetry data captured in production, so eval insights directly translate to production monitoring

    **How It Works**
    When tracing the mlflow experiment, a hash of the input is stored as a span attribute (input_key). The evaluator uses this to find the trace for the given input of the running experiment.

    **Which tools were called** - HasMatchingSpan(query={'name_contains': 'search_tool'})
    **Code paths executed** - Verify specific functions ran or particular branches taken
    **Timing characteristics** - Check that operations complete within SLA bounds
    **Error conditions** - Detect retries, fallbacks, or specific failure modes
    **Execution structure** - Verify parent-child relationships, delegation patterns, or execution order
    This creates a fundamentally different evaluation paradigm: you're testing behavioral contracts, not just input-output relationships.

    """

    _mlflow_settings: MLFlowSettings | None = None
    _mlflow_experiment_id: str | None = None
    _mlflow_run_id: str | None = None
    inputs_to_key_function: Callable[[InputsT], str] = field(default=default_input_to_key, repr=False)

    @property
    def mlflow_settings(self) -> MLFlowSettings:
        if self._mlflow_settings is None:
            self._mlflow_settings = MLFlowSettings()
        return self._mlflow_settings

    @property
    def mlflow_experiment_id(self) -> str:
        if self._mlflow_experiment_id is None:
            experiment = mlflow.get_experiment_by_name(self.mlflow_settings.ragpill_experiment_name)
            if not experiment:
                raise ValueError(f"Experiment {self.mlflow_settings.ragpill_experiment_name} not found.")
            self._mlflow_experiment_id = experiment.experiment_id
        return self._mlflow_experiment_id

    @property
    def mlflow_run_id(self) -> str:
        if self._mlflow_run_id is None:
            if (run := mlflow.active_run()) is None:
                raise ValueError("No active mlflow run found.")
            self._mlflow_run_id = run.info.run_id
        return self._mlflow_run_id

    def get_trace(self, inputs: InputsT) -> Trace:
        # find trace where root span has input_key
        # unforunately, this can't be cashed because for some stupid reason, pydantic-ai is running one input + its evaluators after another,
        # meaning every time this runs for a new input, there's new traces.
        # it could be cashed per input though, which only makes sense if there
        # are a lot of Span-based evaluators per input, which is not the common case, but could be in some scenarios.
        # For now, we keep it simple and don't cache.
        traces = mlflow.search_traces(
            locations=[self.mlflow_experiment_id],
            run_id=self.mlflow_run_id,
            # filter_string=f"span.attributes.input_key LIKE '{self.inputs_to_key_function(inputs)}'",
            return_type="list",
        )
        traces = [
            t
            for t in traces
            if t.data.spans and t.data.spans[0].attributes.get("input_key", "") == self.inputs_to_key_function(inputs)
        ]
        # assert len(traces) == 1, f"Expected exactly one trace for input {inputs}, found {len(traces)}."
        if len(traces) == 0:
            raise ValueError(f"No trace found for input {inputs}.")
        if len(traces) > 1:
            # todo: find right trace (parent)
            return traces[0]  # or raise an error if multiple traces are not expected
        return traces[0]


@dataclass(kw_only=True, repr=False)
class SourcesBaseEvaluator(SpanBaseEvaluator):
    """
    This base class that retrieves the sources from mlflow trace.

    Note: only documents retrieved from a retriever, reranker or tool span are considered as sources.
    """

    evaluation_function: Callable[[list[Document]], bool] = field(repr=False)
    custom_reason_true: str = field(default="Evaluation function returned True.", repr=False)
    custom_reason_false: str = field(default="Evaluation function returned False.", repr=False)

    def get_documents(self, inputs: InputsT) -> list[Document]:
        trace = self.get_trace(inputs)
        retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)
        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
        reranker_spans = trace.search_spans(span_type=SpanType.RERANKER)
        all_documents = []
        for span in retriever_spans + tool_spans + reranker_spans:
            if isinstance(span.outputs, list) and len(span.outputs) > 0:
                try:
                    docs = [
                        Document(**output)
                        for output in span.outputs
                        if isinstance(output, dict) and "page_content" in output and "metadata" in output
                    ]
                except Exception:
                    continue
                all_documents.extend(docs)
        return all_documents

    async def run(
        self,
        ctx: EvaluatorContext[InputsT, OutputT, EvaluatorMetadata],
    ) -> EvaluationReason:
        documents = self.get_documents(ctx.inputs)
        result = self.evaluation_function(documents)
        return EvaluationReason(
            value=result,
            reason=self.custom_reason_true if result else self.custom_reason_false,
        )


def _regex_in_any_document_content(pattern: str) -> Callable[[list[Document]], bool]:
    """Returns a function that checks if the regex pattern is found in any document content."""

    regex = re.compile(pattern)

    def evaluation_function(documents: list[Document]) -> bool:
        for doc in documents:
            normalized_content = _normalize_text(doc.page_content)
            if regex.search(normalized_content):
                return True
        return False

    return evaluation_function


@dataclass(kw_only=True, repr=False)
class RegexInSourcesEvaluator(SourcesBaseEvaluator):
    """
    Evaluator to check if a regex pattern is found in any of the source document's content.
    The documents are retrieved from mlflow trace and include documents from retriever, tool, and reranker spans.

    Tip: Use inline regex flags to modify matching behavior:

    - `(?i)pattern` - Case-insensitive matching (e.g., `(?i)important` matches "Important", "IMPORTANT")
    - `(?s)pattern` - Dotall mode (`.` matches newlines, useful for multi-line content)
    - `(?m)pattern` - Multiline mode (`^` and `$` match line boundaries)
    - `(?ims)pattern` - Combine multiple flags

    Example:
        ```python
        # In CSV testset:
        # check="(?i)section 1"  # Case-insensitive
        # check="(?s)start.*end"  # Match across newlines
        # check="(?i)(?s)important.*conclusion"  # Both case-insensitive and dotall
        ```
    """

    pattern: str

    @classmethod
    def from_csv_line(
        cls, expected: bool, mandatory: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "RegexInSourcesEvaluator":
        """Create a RegexInSourcesEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            mandatory: Whether this evaluation is mandatory
            tags: Comma-separated tags string
            check: Regex pattern to search for in document contents
            **kwargs: Additional attributes for the evaluator
        """
        pattern = _normalize_text(check)
        evaluation_function = _regex_in_any_document_content(pattern)
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            evaluation_function=evaluation_function,
            pattern=pattern,
            attributes=kwargs,
            custom_reason_true=f'Regex pattern "{pattern}" found in at least one document content.',
            custom_reason_false=f'Regex pattern "{pattern}" not found in any document content.',
        )


def _regex_in_doc_metadata(key: str, pattern: str) -> Callable[[list[Document]], bool]:
    """Returns a function that checks if the regex pattern is found in any document metadata value for the given key."""
    regex = re.compile(pattern)

    def evaluation_function(documents: list[Document]) -> bool:
        for doc in documents:
            normalized_metadata_value = _normalize_text(str(doc.metadata.get(key, "")))
            if key in doc.metadata and regex.search(normalized_metadata_value):
                return True
        return False

    return evaluation_function


@dataclass(kw_only=True, repr=False)
class RegexInDocumentMetadataEvaluator(SourcesBaseEvaluator):
    """
    Evaluator to check if a regex pattern is found in a specific metadata field of any
    document retrieved from mlflow trace.

    The documents are retrieved from mlflow trace and include documents from retriever, tool, and reranker spans.

    **Note**:
    For creating from csv, requires 'check' to be a JSON string with 'pattern' and 'key' fields.
    Then checks if any document in the used sources has metadata[key] matching the regex pattern.

    Patterns are normalized to be case-insensitive and Unicode NFKC-cleaned
    by default (e.g., a pattern containing ``UF₆`` will match ``UF6`` in text).
    Inline regex flags still work:

    - `(?s)pattern` - Dotall mode (`.` matches newlines, useful for multi-line metadata values)
    - `(?m)pattern` - Multiline mode (`^` and `$` match line boundaries)
    - `(?ims)pattern` - Combine multiple flags

    Example:
        ```python
        # In CSV testset:
        # check='{"pattern": "(?i)chapter.*", "key": "source"}'  # Case-insensitive match in 'source' metadata
        # check='{"pattern": "(?s)start.*end", "key": "content"}'  # Match across newlines in 'content' metadata
        ```
    """

    metadata_key: str
    pattern: str

    @classmethod
    def from_csv_line(
        cls, expected: bool, mandatory: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "RegexInDocumentMetadataEvaluator":
        """Create a RegexInDocumentMetadataEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            mandatory: Whether this evaluation is mandatory
            tags: Comma-separated tags string
            check: json with 2 keys: "pattern" and "key". Regex pattern to search for in document metadata key.
            **kwargs: Additional attributes for the evaluator
        """
        try:
            check_dict = json.loads(check)
            assert isinstance(check_dict, dict) and "pattern" in check_dict and "key" in check_dict, (
                f"Check must be a JSON object with 'pattern' and 'key'. Got: {check}"
            )
            pattern = check_dict["pattern"]
            metadata_key = check_dict["key"]
        except json.JSONDecodeError:
            raise ValueError(
                f"RegexInDocumentMetadataEvaluator requires 'check' to be a JSON string with 'pattern' and 'key'. But got: {check}"
            )
        pattern = _normalize_text(pattern)
        evaluation_function = _regex_in_doc_metadata(metadata_key, pattern)
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            evaluation_function=evaluation_function,
            metadata_key=metadata_key,
            pattern=pattern,
            attributes=kwargs,
            custom_reason_true=f'Regex pattern "{pattern}" found in key "{metadata_key}" of at least one document content.',
            custom_reason_false=f'Regex pattern "{pattern}" not found in key "{metadata_key}" of any document content.',
        )


@dataclass(kw_only=True, repr=False)
class RegexInOutputEvaluator(BaseEvaluator):
    """Check whether a regex pattern matches the stringified output.

    This evaluator runs the provided regex against ``str(ctx.output)``. Patterns
    are normalized to be case-insensitive and Unicode NFKC-cleaned by default
    (e.g., ``UF₆`` will match ``UF6``). Inline regex flags (e.g., ``(?s)``) are
    supported by the standard ``re`` engine.

    CSV usage examples:
        - ``check="error|failure"``
        - ``check='{"pattern": "(?i)success"}'``
    """

    pattern: str

    def __post_init__(self) -> None:
        try:
            self._compiled_pattern = re.compile(self.pattern)
        except re.error as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid regex pattern '{self.pattern}': {exc}") from exc

    @classmethod
    def from_csv_line(
        cls, expected: bool, mandatory: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "RegexInOutputEvaluator":
        """Create a RegexInOutputEvaluator from a CSV line."""
        if not check or not check.strip():
            raise ValueError("RegexInOutputEvaluator requires a non-empty 'check' pattern.")

        pattern: str = check
        try:
            parsed = json.loads(check)
            if isinstance(parsed, dict) and "pattern" in parsed:
                pattern = str(parsed["pattern"])
            elif isinstance(parsed, str):
                pattern = parsed
        except json.JSONDecodeError:
            pass
        pattern = _normalize_text(pattern)
        return cls(
            pattern=pattern,
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        output_str = str(ctx.output)
        matches = bool(self._compiled_pattern.search(output_str))
        reason = (
            f'Regex pattern "{self.pattern}" matched output.'
            if matches
            else f'Regex pattern "{self.pattern}" did not match output.'
        )
        return EvaluationReason(
            value=matches,
            reason=reason,
        )


@dataclass(kw_only=True, repr=False)
class LiteralQuotationTest(SourcesBaseEvaluator):
    """Verify that all markdown quotes in the output appear literally in source documents.

    This evaluator ensures citations are accurate by checking that any text quoted
    in markdown blockquotes (lines starting with `>`) actually appears in the
    retrieved source documents. This is particularly valuable for RAG systems where
    accuracy of quoted material is critical.

    The evaluator:

    1. Extracts all markdown blockquotes (lines starting with `>`) from the output
    2. Cleans quotes by removing quotation marks and normalizing whitespace
    3. Verifies each quote appears literally (ignoring whitespace) in source documents
    4. Reports any missing quotes with their referenced filenames when available

    Only lines starting with `>` (after leading whitespace) are considered markdown
    quotes. Regular quoted text like `"this"` or `'this'` is ignored.

    Args:
        expected: Expected evaluation result (default: True)
        mandatory: Whether this evaluation is mandatory (default: True)
        tags: Set of tags for categorizing this evaluator
        attributes: Additional attributes for the evaluator

    Example:
        ```python
        from ragpill.evaluators import LiteralQuotationTest

        # Create evaluator
        evaluator = LiteralQuotationTest(
            expected=True,
            mandatory=True,
            tags={"quotation", "accuracy"}
        )

        # Output with markdown quote
        output = '''
        The report states:
        > "'no longer outstanding at this stage' does not mean 'resolved'."
        (File: [report.txt](link), Paragraph: 38)
        '''

        # The evaluator will verify this quote exists in the source documents
        ```

    Markdown Quote Format:
        The evaluator recognizes standard markdown blockquotes:

        ```markdown
        > This is a single-line quote

        > This is a multi-line quote
        > that continues on the next line

        > Quote with file reference
        (File: [document.txt](link), Paragraph: 5)
        ```

    Note:
        - Whitespace differences between quotes and source text are ignored
        - Quotation marks (`"`, `'`, `'`, `'`, `"`, `"`) are stripped before comparison
        - File references in format `(File: [filename](...))` are extracted and included in error messages
        - Empty quotes (after cleaning) are skipped
        - Quotes must appear literally in source documents (no fuzzy matching)

    See Also:
        [`SourcesBaseEvaluator`][ragpill.evaluators.SourcesBaseEvaluator]:
            Base class that retrieves source documents from MLflow traces
        [`RegexInSourcesEvaluator`][ragpill.evaluators.RegexInSourcesEvaluator]:
            Similar evaluator using regex patterns instead of literal quotes
    """

    def __init__(
        self,
        expected: bool = True,
        mandatory: bool = True,
        tags: set[str] | None = None,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            evaluation_function=lambda docs: True,  # Placeholder, actual logic is in run() for access to output
            expected=expected,
            mandatory=mandatory,
            tags=tags or set(),
            attributes=attributes or {},
            custom_reason_true="All quotes found in source documents.",
            custom_reason_false="",  # Will be set dynamically
            **kwargs,
        )

    @classmethod
    def from_csv_line(
        cls, expected: bool, mandatory: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "LiteralQuotationTest":
        """Create a LiteralQuotationTest from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            mandatory: Whether this evaluation is mandatory
            tags: Comma-separated tags string
            check: Not used for this evaluator (can be empty)
            **kwargs: Additional attributes for the evaluator
        """
        return cls(
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[InputsT, OutputT, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Override run to have access to both output and documents."""
        documents = self.get_documents(ctx.inputs)
        output_str = str(ctx.output)

        # Extract normalized quotes from output
        quotes = _extract_markdown_quotes(output_str)

        if not quotes:
            return EvaluationReason(
                value=True,
                reason="No quotes found in output.",
            )

        # Normalize all document contents
        normalized_docs = [_normalize_text(doc.page_content) for doc in documents]

        # Check each quote
        not_found = []
        for quote, referenced_file in quotes:
            # Check if quote appears in any document
            # Use regex search if quote contains .* (from ellipsis conversion), otherwise use substring match
            if ".*" in quote:
                pattern = re.escape(quote).replace(r"\.\*", ".*")
                found = any(re.search(pattern, doc_content) for doc_content in normalized_docs)
            else:
                found = any(quote in doc_content for doc_content in normalized_docs)

            if not found:
                if referenced_file:
                    not_found.append(f'"{quote}" (Referenced file: {referenced_file})')
                else:
                    not_found.append(f'"{quote}"')

        if not_found:
            reason = f"Quotes not found in sources: {'; '.join(not_found)}"
            return EvaluationReason(
                value=False,
                reason=reason,
            )

        return EvaluationReason(
            value=True,
            reason=f"All {len(quotes)} quote(s) found in source documents.",
        )


@dataclass(kw_only=True, repr=False)
class HasQuotesEvaluator(BaseEvaluator):
    """Check if the output contains a minimum (and optionally maximum) number of markdown quotes.

    This evaluator verifies that the output includes at least a specified number
    of markdown blockquotes (lines starting with `>`). Useful for ensuring responses
    include citations, evidence, or quoted material.

    Only lines starting with `>` (after leading whitespace) are considered markdown
    quotes. Regular quoted text like `"this"` or `'this'` is ignored.

    Args:
        min_quotes: Minimum number of quotes required (default: 1)
        max_quotes: Maximum number of quotes allowed (default: -1, meaning no maximum)
        expected: Expected evaluation result (default: True)
        mandatory: Whether this evaluation is mandatory (default: True)
        tags: Set of tags for categorizing this evaluator
        attributes: Additional attributes for the evaluator

    Example:
        ```python
        from ragpill.evaluators import HasQuotesEvaluator

        # Require at least 2 quotes
        evaluator = HasQuotesEvaluator(
            min_quotes=2,
            expected=True,
            mandatory=True,
            tags={"quotation", "format"}
        )

        # Require between 2 and 5 quotes
        evaluator = HasQuotesEvaluator(
            min_quotes=2,
            max_quotes=5,
            expected=True,
            mandatory=True,
            tags={"quotation", "format"}
        )

        # This output has 2 quotes and will pass
        output = '''
        The report states two key points:
        > "First important point."

        And also:
        > "Second important point."
        '''
        ```

    Note:
        - Multi-line quotes (consecutive lines with `>`) are counted as one quote
        - Empty quotes (only whitespace after `>`) are not counted
        - The evaluator passes if min_quotes <= num_quotes <= max_quotes (or no max if max_quotes=-1)
        - Set expected=False to verify that quotes are NOT within the specified range

    See Also:
        [`LiteralQuotationTest`][ragpill.evaluators.LiteralQuotationTest]:
            Verifies quotes appear literally in source documents
        [`BaseEvaluator`][ragpill.base.BaseEvaluator]:
            Base class for all evaluators
    """

    min_quotes: int = field(default=1)
    max_quotes: int = field(default=-1)

    @classmethod
    def from_csv_line(
        cls, expected: bool, mandatory: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "HasQuotesEvaluator":
        """Create a HasQuotesEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            mandatory: Whether this evaluation is mandatory
            tags: Comma-separated tags string
            check: Either an integer for min_quotes, or JSON with 'min_quotes' and optionally 'max_quotes'.
                   If empty, defaults to min_quotes=1, max_quotes=-1.
            **kwargs: Additional attributes for the evaluator

        Example:
            In CSV, use check="3" to require at least 3 quotes.
            Or use check='{"min_quotes": 2, "max_quotes": 5}' to require 2-5 quotes.
        """
        min_quotes = 1  # default
        max_quotes = -1  # default (no maximum)

        if check and check.strip():
            # Try parsing as JSON first
            try:
                check_dict = json.loads(check)
                if isinstance(check_dict, dict):
                    min_quotes = check_dict.get("min_quotes", min_quotes)
                    max_quotes = check_dict.get("max_quotes", max_quotes)
                elif isinstance(check_dict, (int, float)):
                    min_quotes = int(check_dict)
                else:
                    raise ValueError("JSON must be an object or number")

                if min_quotes < 0:
                    raise ValueError(f"min_quotes must be non-negative, got {min_quotes}")
                if max_quotes != -1 and max_quotes < min_quotes:
                    raise ValueError(f"max_quotes ({max_quotes}) must be >= min_quotes ({min_quotes}) or -1")
            except json.JSONDecodeError:
                # Not JSON, treat as integer for min_quotes
                try:
                    min_quotes = int(check)
                    if min_quotes < 0:
                        raise ValueError(f"min_quotes must be non-negative, got {min_quotes}")
                except ValueError as e:
                    raise ValueError(
                        f"HasQuotesEvaluator 'check' parameter must be a non-negative integer or JSON object. Got: {check}"
                    ) from e

        return cls(
            min_quotes=min_quotes,
            max_quotes=max_quotes,
            expected=expected,
            mandatory=mandatory,
            tags=tags,
            attributes=kwargs,
        )

    def _extract_quotes_from_output(self, output: str) -> list[str]:
        """Extract markdown quotes from output.

        Only lines that start with '>' (after leading whitespace) are considered markdown quotes.

        Returns:
            List of quote texts (after cleaning)
        """
        # Use shared function and discard file references
        return [quote for quote, _ in _extract_markdown_quotes(output)]

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Check if output contains the required number of quotes (within min/max bounds)."""
        output_str = str(ctx.output)

        # Extract quotes from output
        quotes = self._extract_quotes_from_output(output_str)
        num_quotes = len(quotes)

        # Check if we have enough quotes
        has_min = num_quotes >= self.min_quotes
        has_max = self.max_quotes == -1 or num_quotes <= self.max_quotes
        passes = has_min and has_max

        # Build reason message
        if passes:
            if self.max_quotes == -1:
                reason = f"Found {num_quotes} quote(s) in output (minimum required: {self.min_quotes})."
            else:
                reason = f"Found {num_quotes} quote(s) in output (required range: {self.min_quotes}-{self.max_quotes})."
        else:
            if not has_min:
                reason = f"Found only {num_quotes} quote(s) in output, but {self.min_quotes} required."
            else:  # not has_max
                reason = f"Found {num_quotes} quote(s) in output, but maximum allowed is {self.max_quotes}."

        return EvaluationReason(
            value=passes,
            reason=reason,
        )
