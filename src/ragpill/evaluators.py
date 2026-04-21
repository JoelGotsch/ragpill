import json
import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities import Document, SpanType, Trace
from pydantic_ai import models
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output, judge_output

from ragpill.base import (
    BaseEvaluator,
    EvaluatorMetadata,
    _current_run_span_id,  # pyright: ignore[reportPrivateUsage]
    default_input_to_key,
)
from ragpill.settings import MLFlowSettings, get_llm_judge_settings
from ragpill.utils import (
    _extract_markdown_quotes,  # pyright: ignore[reportPrivateUsage]
    _normalize_text,  # pyright: ignore[reportPrivateUsage]
)


def _get_default_judge_llm() -> models.Model:
    """Get default LLM model from the global :class:`LLMJudgeSettings` singleton."""
    return get_llm_judge_settings().llm_model


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
        rubric: str = check
        try:
            check_obj: Any = json.loads(check)
            if isinstance(check_obj, dict):
                rubric = str(check_obj.pop("rubric", check))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                kwargs.update(check_obj)  # pyright: ignore[reportUnknownArgumentType]
        except json.JSONDecodeError:
            # Plain text - use as rubric
            pass
        model = kwargs.pop("model", None) or get_llm()

        return cls(
            rubric=rubric,
            model=model,
            expected=expected,
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Evaluate the output against the rubric using an LLM judge.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result with the judge's reasoning.
        """
        # Wrap in an explicit span so that both the pydantic-ai and openai autolog
        # integrations create child spans rather than competing root traces. Without this,
        # the two integrations race to INSERT a root trace with the same request_id, which
        # causes a UNIQUE constraint violation in MLflow's SQLite backend.
        # The "ragpill_is_judge_trace" attribute lets _delete_llm_judge_traces identify
        # and remove these traces after evaluation.
        with mlflow.start_span(name="llm-judge-evaluation", span_type=SpanType.LLM) as span:
            span.set_attribute("ragpill_is_judge_trace", True)
            if self.include_input:
                grading_output = await judge_input_output(ctx.inputs, ctx.output, self.rubric, self.model)
            else:
                grading_output = await judge_output(ctx.output, self.rubric, self.model)
            span.set_outputs({"pass": grading_output.pass_, "reason": grading_output.reason})
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
            attributes=self.attributes,
            tags=self.tags,
            is_global_evaluator=self.is_global,
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
            tags={"tag1", "tag2"},
            attributes={"attr1": "value1"},
        )
    """

    pydantic_evaluator: Evaluator = field(repr=False)

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Delegate evaluation to the wrapped pydantic-evals evaluator.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result from the wrapped evaluator.
        """
        eval_output: Any = await self.pydantic_evaluator.evaluate(ctx)  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]
        assert isinstance(eval_output, EvaluationReason), (
            "Wrapped pydantic evaluator did not return an EvaluationReason."
        )
        return eval_output


def _filter_trace_to_subtree(trace: Trace, root_span_id: str) -> Trace:
    """Return a copy of the trace containing only the subtree rooted at root_span_id.

    This ensures span-based evaluators only see spans from their specific run,
    not spans from other runs in the same case trace.

    Args:
        trace: The full trace to filter.
        root_span_id: The span ID of the subtree root.

    Returns:
        A new Trace with only the matching subtree spans.
    """
    all_spans = trace.data.spans
    included: set[str] = set()
    queue = [root_span_id]
    while queue:
        current = queue.pop()
        included.add(current)
        for span in all_spans:
            if span.parent_id == current:
                queue.append(span.span_id)
    filtered_data = copy(trace.data)
    filtered_data.spans = [s for s in all_spans if s.span_id in included]
    return Trace(info=trace.info, data=filtered_data)


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
    inputs_to_key_function: Callable[[Any], str] = field(default=default_input_to_key, repr=False)

    @property
    def mlflow_settings(self) -> MLFlowSettings:
        if self._mlflow_settings is None:
            self._mlflow_settings = MLFlowSettings()  # pyright: ignore[reportCallIssue]
        return self._mlflow_settings

    @property
    def mlflow_experiment_id(self) -> str:
        if self._mlflow_experiment_id is None:
            experiment = mlflow.get_experiment_by_name(self.mlflow_settings.ragpill_experiment_name)
            if not experiment or not experiment.experiment_id:
                raise ValueError(f"Experiment {self.mlflow_settings.ragpill_experiment_name} not found.")
            self._mlflow_experiment_id = experiment.experiment_id
        result = self._mlflow_experiment_id
        assert result is not None
        return result

    @property
    def mlflow_run_id(self) -> str:
        if self._mlflow_run_id is None:
            run = mlflow.active_run()
            if run is None:
                raise ValueError("No active mlflow run found.")
            self._mlflow_run_id = run.info.run_id
        result = self._mlflow_run_id
        assert result is not None
        return result

    def get_trace(self, inputs: Any) -> Trace:
        """Find the MLflow trace for the given inputs and optionally filter to the current run's subtree.

        When ``_current_run_span_id`` is set (during multi-run evaluation Phase 2),
        the returned trace is filtered to only contain spans from the current run's subtree.
        This prevents evaluators from accidentally inspecting spans from other runs.
        """
        target_key = self.inputs_to_key_function(inputs)
        traces: list[Trace] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
            locations=[self.mlflow_experiment_id],
            run_id=self.mlflow_run_id,
            return_type="list",
        )
        # Match traces by input_key on any span (not just root), since in multi-run
        # mode the root span is the case-level parent and run spans are children.
        matching: list[Trace] = []
        for t in traces:
            for span in t.data.spans or []:
                if span.attributes.get("input_key", "") == target_key:
                    matching.append(t)
                    break
        if len(matching) == 0:
            raise ValueError(f"No trace found for input {inputs}.")

        trace = matching[0]

        # If we're inside a multi-run evaluation, filter to only the current run's subtree.
        run_span_id = _current_run_span_id.get()
        if run_span_id is not None:
            trace = _filter_trace_to_subtree(trace, run_span_id)

        return trace


@dataclass(kw_only=True, repr=False)
class SourcesBaseEvaluator(SpanBaseEvaluator):
    """
    This base class that retrieves the sources from mlflow trace.

    Note: only documents retrieved from a retriever, reranker or tool span are considered as sources.
    """

    evaluation_function: Callable[[list[Document]], bool] = field(repr=False)
    custom_reason_true: str = field(default="Evaluation function returned True.", repr=False)
    custom_reason_false: str = field(default="Evaluation function returned False.", repr=False)

    def get_documents(self, inputs: Any) -> list[Document]:
        """Retrieve source documents from MLflow trace spans.

        Searches retriever, tool, and reranker spans in the trace for the given
        inputs and collects all documents found in their outputs.

        Args:
            inputs: The task inputs used to look up the corresponding trace.

        Returns:
            List of documents extracted from the trace spans.
        """
        trace = self.get_trace(inputs)
        retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)  # pyright: ignore[reportArgumentType,reportUnknownMemberType]
        tool_spans = trace.search_spans(span_type=SpanType.TOOL)  # pyright: ignore[reportArgumentType,reportUnknownMemberType]
        reranker_spans = trace.search_spans(span_type=SpanType.RERANKER)  # pyright: ignore[reportArgumentType,reportUnknownMemberType]
        all_documents: list[Document] = []
        for span in retriever_spans + tool_spans + reranker_spans:
            if isinstance(span.outputs, list) and len(span.outputs) > 0:  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                try:
                    docs = [
                        Document(**output)  # pyright: ignore[reportUnknownArgumentType]
                        for output in span.outputs  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                        if isinstance(output, dict) and "page_content" in output and "metadata" in output
                    ]
                except Exception:
                    continue
                all_documents.extend(docs)
        return all_documents

    async def run(
        self,
        ctx: EvaluatorContext[Any, Any, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Retrieve source documents and apply the evaluation function.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result with a custom reason message.
        """
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

    Both the pattern and document contents are normalized before matching via
    ``_normalize_text``, which applies:

    - **Case-folding** - all text is lowercased (``str.casefold``), so matching
      is always case-insensitive. Using the ``(?i)`` flag is therefore redundant.
    - **Unicode NFKC** - compatibility characters are unified
      (e.g. ``UF₆`` ↔ ``UF6``).
    - **Whitespace collapsing** - runs of whitespace become a single space.
    - **Quote normalization** - curly quotes, guillemets, primes, etc. are
      replaced with a straight single quote ``'``.
    - **Markdown subscript stripping** - e.g. ``UF~6~`` → ``UF6``.
    - **Trailing period stripping**.

    Tip: Use inline regex flags to modify matching behavior:

    - `(?s)pattern` - Dotall mode (`.` matches newlines, useful for multi-line content)
    - `(?m)pattern` - Multiline mode (`^` and `$` match line boundaries)
    - `(?ms)pattern` - Combine multiple flags

    Example:
        ```python
        # In CSV testset:
        # check="section 1"  # Already case-insensitive via normalization
        # check="(?s)start.*end"  # Match across newlines
        # check="(?s)important.*conclusion"  # Dotall for multi-line matching
        ```
    """

    pattern: str

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> "RegexInSourcesEvaluator":
        """Create a RegexInSourcesEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            tags: Comma-separated tags string
            check: Regex pattern to search for in document contents
            **kwargs: Additional attributes for the evaluator
        """
        pattern = _normalize_text(check)
        evaluation_function = _regex_in_any_document_content(pattern)
        return cls(
            expected=expected,
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

    Both the pattern and metadata values are normalized before matching via
    ``_normalize_text``, which applies case-folding (``str.casefold``),
    Unicode NFKC, whitespace collapsing, and quote normalization. Because text
    is already case-folded, the ``(?i)`` flag is redundant.

    Inline regex flags still work:

    - `(?s)pattern` - Dotall mode (`.` matches newlines, useful for multi-line metadata values)
    - `(?m)pattern` - Multiline mode (`^` and `$` match line boundaries)
    - `(?ms)pattern` - Combine multiple flags

    Example:
        ```python
        # In CSV testset:
        # check='{"pattern": "chapter.*", "key": "source"}'  # Already case-insensitive via normalization
        # check='{"pattern": "(?s)start.*end", "key": "content"}'  # Match across newlines in 'content' metadata
        ```
    """

    metadata_key: str
    pattern: str

    @classmethod
    def from_csv_line(
        cls, expected: bool, tags: set[str], check: str, **kwargs: Any
    ) -> "RegexInDocumentMetadataEvaluator":
        """Create a RegexInDocumentMetadataEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            tags: Comma-separated tags string
            check: json with 2 keys: "pattern" and "key". Regex pattern to search for in document metadata key.
            **kwargs: Additional attributes for the evaluator
        """
        try:
            check_dict: Any = json.loads(check)
            assert isinstance(check_dict, dict) and "pattern" in check_dict and "key" in check_dict, (
                f"Check must be a JSON object with 'pattern' and 'key'. Got: {check}"
            )
            pattern: str = str(check_dict["pattern"])  # pyright: ignore[reportUnknownArgumentType]
            metadata_key: str = str(check_dict["key"])  # pyright: ignore[reportUnknownArgumentType]
        except json.JSONDecodeError:
            raise ValueError(
                f"RegexInDocumentMetadataEvaluator requires 'check' to be a JSON string with 'pattern' and 'key'. But got: {check}"
            )
        pattern = _normalize_text(pattern)
        evaluation_function = _regex_in_doc_metadata(metadata_key, pattern)
        return cls(
            expected=expected,
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

    Both the pattern and the output are normalized before matching via
    ``_normalize_text``, which applies case-folding (``str.casefold``),
    Unicode NFKC, whitespace collapsing, and quote normalization.
    Because text is already case-folded, the ``(?i)`` flag is redundant.

    CSV usage examples:
        - ``check="error|failure"``
        - ``check='{"pattern": "success"}'``
    """

    pattern: str

    def __post_init__(self) -> None:
        try:
            self._compiled_pattern = re.compile(self.pattern)
        except re.error as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid regex pattern '{self.pattern}': {exc}") from exc

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> "RegexInOutputEvaluator":
        """Create a RegexInOutputEvaluator from a CSV line."""
        if not check or not check.strip():
            raise ValueError("RegexInOutputEvaluator requires a non-empty 'check' pattern.")

        pattern: str = check
        try:
            parsed: dict[str, Any] | str = json.loads(check)
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
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[object, object, EvaluatorMetadata],
    ) -> EvaluationReason:
        """Check whether the regex pattern matches the normalized task output.

        Args:
            ctx: The evaluator context containing inputs, output, and metadata.

        Returns:
            The evaluation result indicating whether the pattern matched.
        """
        output_str = _normalize_text(str(ctx.output))
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
class LiteralQuoteEvaluator(SourcesBaseEvaluator):
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
        tags: Set of tags for categorizing this evaluator
        attributes: Additional attributes for the evaluator

    Example:
        ```python
        from ragpill.evaluators import LiteralQuoteEvaluator

        # Create evaluator
        evaluator = LiteralQuoteEvaluator(
            expected=True,
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
        tags: set[str] | None = None,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            evaluation_function=lambda docs: True,  # Placeholder, actual logic is in run() for access to output
            expected=expected,
            tags=tags or set(),
            attributes=attributes or {},
            custom_reason_true="All quotes found in source documents.",
            custom_reason_false="",  # Will be set dynamically
            **kwargs,
        )

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> "LiteralQuoteEvaluator":
        """Create a LiteralQuoteEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
            tags: Comma-separated tags string
            check: Not used for this evaluator (can be empty)
            **kwargs: Additional attributes for the evaluator
        """
        return cls(
            expected=expected,
            tags=tags,
            attributes=kwargs,
        )

    async def run(
        self,
        ctx: EvaluatorContext[Any, Any, EvaluatorMetadata],
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
        not_found: list[str] = []
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
        tags: Set of tags for categorizing this evaluator
        attributes: Additional attributes for the evaluator

    Example:
        ```python
        from ragpill.evaluators import HasQuotesEvaluator

        # Require at least 2 quotes
        evaluator = HasQuotesEvaluator(
            min_quotes=2,
            expected=True,
            tags={"quotation", "format"}
        )

        # Require between 2 and 5 quotes
        evaluator = HasQuotesEvaluator(
            min_quotes=2,
            max_quotes=5,
            expected=True,
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
        [`LiteralQuoteEvaluator`][ragpill.evaluators.LiteralQuoteEvaluator]:
            Verifies quotes appear literally in source documents
        [`BaseEvaluator`][ragpill.base.BaseEvaluator]:
            Base class for all evaluators
    """

    min_quotes: int = field(default=1)
    max_quotes: int = field(default=-1)

    @classmethod
    def from_csv_line(cls, expected: bool, tags: set[str], check: str, **kwargs: Any) -> "HasQuotesEvaluator":
        """Create a HasQuotesEvaluator from a CSV line.

        This method is used by the CSV testset loader to instantiate the evaluator.
        See [`load_testset`][ragpill.csv.testset.load_testset] for more details.

        Args:
            expected: Expected evaluation result
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
                check_parsed: Any = json.loads(check)
                if isinstance(check_parsed, dict):
                    min_quotes = int(check_parsed.get("min_quotes", min_quotes))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    max_quotes = int(check_parsed.get("max_quotes", max_quotes))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                elif isinstance(check_parsed, (int, float)):
                    min_quotes = int(check_parsed)
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
