import asyncio
import concurrent.futures
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, Experiment, Feedback, SpanType, Trace
from pydantic import TypeAdapter
from pydantic_evals import Dataset
from pydantic_evals.evaluators.evaluator import EvaluationResult, EvaluatorSpec
from pydantic_evals.reporting import EvaluationReport, ReportCase

from ragpill.base import (
    BaseEvaluator,
    CaseMetadataT,
    EvaluatorMetadata,
    TestCaseMetadata,
    default_input_to_key,
    merge_metadata,
)
from ragpill.settings import MLFlowSettings
from ragpill.utils import _fix_evaluator_global_flag  # pyright: ignore[reportPrivateUsage]

TaskType = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]


def _mlflow_runnable_wrapper(
    task: TaskType,
    input_to_key: Callable[[Any], str] = default_input_to_key,
    # input_parent_span_id: dict | None = None,
) -> TaskType:
    """Wrap a function to be runnable asynchronously for mlflow tracing. Make sure that mlflow is logging (mlflow.autolog() or similar) and mlflow.set_experiment(<your-experiment-name>) are set before executing this wrapper."""
    # input_parent_span_id = input_parent_span_id or {}
    # mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
    # mlflow.set_experiment(mlflow_settings.experiment_name)

    if inspect.iscoroutinefunction(task):

        async def async_runnable(input: Any) -> Any:
            with mlflow.start_span(name="test-span", span_type=SpanType.TASK) as span:
                span.set_inputs(input)
                input_key = input_to_key(input)
                span.set_attribute("input_key", input_key)
                # input_parent_span_id[input_key] = span.span_id
                output = await task(input)
                span.set_outputs(output)
                return output

        return async_runnable
    else:

        def sync_runnable(input: Any) -> Any:
            with mlflow.start_span(name="test-span", span_type=SpanType.TASK) as span:
                span.set_inputs(input)
                input_key = input_to_key(input)
                span.set_attribute("input_key", input_key)
                # input_parent_span_id[input_key] = span.span_id
                output = task(input)
                span.set_outputs(output)
                return output

        return sync_runnable


def _setup_mlflow_experiment(mlflow_settings: MLFlowSettings) -> None:
    """Setup mlflow experiment with given settings."""
    mlflow.set_tracking_uri(mlflow_settings.ragpill_tracking_uri)
    mlflow.set_experiment(mlflow_settings.ragpill_experiment_name)  # pyright: ignore[reportUnknownMemberType]
    # mlflow.autolog()  # should cover pydantic-ai and openai
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]  # pydantic-ai and openai are racing each other
    mlflow.start_run(description=mlflow_settings.ragpill_run_description)


def _delete_llm_judge_traces(mlflow_settings: MLFlowSettings) -> tuple[Experiment, str]:
    """
    LLMJudge produces traces for each evaluation, which clutters the mlflow tracing UI.
    We want only traces that relate to the actual task execution, not the evaluation of the outputs of the task-runs.

    :param mlflow_settings: The mlflow settings to use.
    :type mlflow_settings: MLFlowSettings
    """
    experiment = mlflow.get_experiment_by_name(mlflow_settings.ragpill_experiment_name)
    assert experiment is not None, f"Experiment '{mlflow_settings.ragpill_experiment_name}' not found."
    experiment_id: str = str(experiment.experiment_id)  # pyright: ignore[reportUnknownArgumentType]
    df: Any = mlflow.search_runs([experiment_id], order_by=["start_time DESC"])
    latest_run_id: str = str(df.iloc[0]["run_id"])

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_settings.ragpill_tracking_uri)
    traces: list[Trace] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
        locations=[experiment_id], run_id=latest_run_id, return_type="list"
    )
    delete_traces: list[str] = []
    for trace in traces:
        root = trace.data._get_root_span()  # pyright: ignore[reportPrivateUsage]
        if root and root.attributes.get("ragpill_is_judge_trace"):
            delete_traces.append(trace.info.trace_id)
    if delete_traces:
        client.delete_traces(experiment_id=experiment_id, trace_ids=delete_traces)
    return experiment, latest_run_id


def _get_input_key_trace_id_map(experiment: Experiment, latest_run_id: str) -> dict[str, str]:
    traces: list[Trace] = mlflow.search_traces(  # pyright: ignore[reportAssignmentType]
        locations=[experiment.experiment_id], run_id=latest_run_id, return_type="list"
    )
    input_key_trace_map: dict[str, str] = {}
    for trace in traces:
        span = trace.data._get_root_span()  # pyright: ignore[reportPrivateUsage]
        if not span:
            continue
        input_key = span.attributes.get("input_key")
        if not input_key:
            # Non-task traces (e.g. LLMJudge evaluation traces) have no input_key; skip them.
            # _delete_llm_judge_traces should have removed these already, but we guard here
            # in case deletion failed or a third-party trace ended up in the same run.
            continue
        input_key_trace_map[input_key] = trace.info.trace_id
    return input_key_trace_map


def _get_input_key_report_case_map(
    testsetresults: EvaluationReport, testset: Dataset[Any, Any, CaseMetadataT]
) -> dict[str, ReportCase]:
    input_key_report_case_map: dict[str, ReportCase] = {}
    for case in testsetresults.cases:
        input_key = default_input_to_key(case.inputs)
        assert (
            input_key not in input_key_report_case_map
        ), f"Duplicate input_key found: {input_key}. Please only create one Case Per Input."
        input_key_report_case_map[input_key] = case
    # handle task failures (error in task execution, not evaluation failure, those are handled in the assertions of each case and will have an assertion with value False and reason describing the error. here we want to catch cases where the task execution itself failed and no assertions were produced, so we can log those properly in mlflow as well.):
    # task failures should be handled such that all evaluators on it are marked as failed, the output
    input_key_report_case_map |= _handle_task_failures(testsetresults, testset)
    return input_key_report_case_map


def _get_evaluation_id_eval_metadata_map(
    testset: Dataset[Any, Any, CaseMetadataT],
) -> dict[str, EvaluatorMetadata]:
    eval_metadata_map: dict[str, EvaluatorMetadata] = {}
    for case in testset.cases:
        input_key = default_input_to_key(case.inputs)
        for evaluator in case.evaluators:
            assert isinstance(
                evaluator, BaseEvaluator
            ), "Only BaseEvaluator derived evaluators are supported in this logging script."
            eval_metadata_map[f"{input_key}_{evaluator.evaluation_name}"] = evaluator.metadata
        for evaluator in testset.evaluators:
            assert isinstance(
                evaluator, BaseEvaluator
            ), "Only BaseEvaluator derived evaluators are supported in this logging script."
            eval_metadata_map[f"{input_key}_{evaluator.evaluation_name}"] = evaluator.metadata

    return eval_metadata_map


def _handle_task_failures(
    testsetresults: EvaluationReport, dataset: Dataset[Any, Any, CaseMetadataT]
) -> dict[str, ReportCase]:
    input_key_failed_report_case_map: dict[str, ReportCase] = {}
    failed_evaluators: dict[str, list[Any]] = {default_input_to_key(c.inputs): c.evaluators for c in dataset.cases}
    for failed_case in testsetresults.failures:
        input_key = default_input_to_key(failed_case.inputs)
        evaluators = failed_evaluators.get(input_key, [])
        assertions: dict[str, EvaluationResult] = {}
        for evaluator in evaluators:
            assert isinstance(
                evaluator, BaseEvaluator
            ), "Only BaseEvaluator derived evaluators are supported in this logging script."
            evaluator_name = evaluator.get_serialization_name()
            if (
                evaluator_name in assertions
            ):  # change to f"{evaluator_name}_{n+1}" where n is number of times this evaluator is in already
                n = len([k for k in assertions.keys() if k.startswith(evaluator_name)])
                evaluator_name = f"{evaluator_name}_{n + 1}"
            assert (
                evaluator_name not in assertions
            ), f"Duplicate evaluator name found for failed case: {evaluator_name}. Please make sure each evaluator has a unique name."
            assertions[evaluator_name] = EvaluationResult(
                name=evaluator_name,
                value=False,
                reason="Task execution failed, evaluation defaults to failed.",
                source=EvaluatorSpec(name="CODE", arguments={"evaluation_name": evaluator.evaluation_name}),
            )
        assert isinstance(
            failed_case.metadata, TestCaseMetadata
        ), "TestCaseMetadata is required in ReportCase metadata for task failure cases."
        input_key_failed_report_case_map[input_key] = ReportCase(
            name=failed_case.name,
            inputs=failed_case.inputs,
            output=f"Task execution failed with error: {failed_case.error_message}\n\n{failed_case.error_stacktrace}",
            assertions=assertions,  # pyright: ignore[reportArgumentType]
            metadata=failed_case.metadata,
            trace_id=failed_case.trace_id,
            span_id=failed_case.span_id,
            # 'expected_output', 'metrics', 'attributes', 'scores', 'labels', 'task_duration', and 'total_duration'
            expected_output=None,
            metrics={},
            attributes={},
            scores={},
            labels=failed_case.metadata.tags,  # pyright: ignore[reportArgumentType]
            task_duration=0.0,
            total_duration=0.0,
        )
    return input_key_failed_report_case_map


ta = TypeAdapter(dict[str, Any])


# TODO: decouple mlflow stuff from the dataframe creation
# use dataframe to log to mlflow
def _create_evaluation_dataframe(
    input_key_trace_map: dict[str, str],
    input_key_report_case_map: dict[str, ReportCase],
    eval_metadata_map: dict[str, EvaluatorMetadata],
) -> pd.DataFrame:
    df_rows: list[dict[str, Any]] = []
    for input_key, trace_id in input_key_trace_map.items():
        reportcase = input_key_report_case_map[input_key]
        assert isinstance(reportcase.metadata, TestCaseMetadata), "ReportCase metadata is not of type TestCaseMetadata."
        for evaluator_name, eval_result in reportcase.assertions.items():
            evaluator_metadata = eval_metadata_map.get(
                f"{input_key}_{eval_result.source.arguments.get('evaluation_name')}"  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportUnknownMemberType]
            )
            assert isinstance(evaluator_metadata, EvaluatorMetadata), "Evaluator metadata not found for evaluator."
            # case_metadata =
            merged_metadata = merge_metadata(reportcase.metadata, evaluator_metadata)
            # source type can either be LLM_JUDGE or CODE (or HUMAN)
            source_type = "LLM_JUDGE" if "LLMJudge" in eval_result.source.name else "CODE"
            df_rows.append(
                {
                    "inputs": str(reportcase.inputs),
                    "output": str(reportcase.output),
                    "evaluator_result": eval_result.value,
                    "evaluator_data": merged_metadata.other_evaluator_data,
                    "evaluator_reason": eval_result.reason,
                    "expected": merged_metadata.expected,
                    "attributes": ta.dump_json(merged_metadata.attributes),
                    "tags": merged_metadata.tags,
                    "task_duration": reportcase.task_duration,
                    "evaluator_name": evaluator_name,
                    "case_name": reportcase.name,
                    "case_id": input_key,
                    "source_type": source_type,
                    "source_id": eval_result.source.name,
                    "input_key": input_key,
                    "trace_id": trace_id,
                }
            )
        # handle evaluator failures (errors in evaluation code)
        for eval_fails in reportcase.evaluator_failures:
            evaluator_metadata = eval_metadata_map.get(
                f"{input_key}_{eval_fails.source.arguments.get('evaluation_name')}"  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportUnknownMemberType]
            )
            assert isinstance(evaluator_metadata, EvaluatorMetadata), "Evaluator metadata not found for evaluator."
            merged_metadata = merge_metadata(reportcase.metadata, evaluator_metadata)
            df_rows.append(
                {
                    "inputs": str(reportcase.inputs),
                    "output": str(reportcase.output),
                    "evaluator_result": False,
                    "evaluator_data": merged_metadata.other_evaluator_data,
                    "evaluator_reason": f"Evaluator failed with error: {eval_fails.error_message}\n\n{eval_fails.error_stacktrace}",
                    "expected": merged_metadata.expected,
                    "attributes": ta.dump_json(merged_metadata.attributes),
                    "tags": merged_metadata.tags,
                    "task_duration": reportcase.task_duration,
                    "evaluator_name": eval_fails.name,
                    "case_name": reportcase.name,
                    "case_id": input_key,
                    "source_type": "CODE",
                    "source_id": eval_fails.source.name,
                    "input_key": input_key,
                    "trace_id": trace_id,
                }
            )
    return pd.DataFrame(df_rows)


def _upload_mlflow(
    eval_result_df: pd.DataFrame,
    input_key_report_case_map: dict[str, ReportCase],
    model_params: dict[str, str] | None = None,
) -> None:
    """Upload evaluation results to mlflow as metrics and params.

    :param eval_result_df: The evaluation results dataframe.
    :type eval_result_df: pd.DataFrame
    """
    mlflow.log_table(eval_result_df, "evaluation_results.json")
    # log model params if any
    if model_params:
        mlflow.log_params(model_params)
    # log evaluation results for all assessments

    # Calculate overall accuracy (assuming evaluator_result is boolean or 0/1)
    # Filter out None/NaN values before calculating
    # pandas typing is incomplete; cast to Any for intermediate DataFrame operations
    eval_df: Any = eval_result_df
    df_valid = eval_df[eval_df["evaluator_result"].notna()]
    overall_accuracy: float = float(df_valid["evaluator_result"].mean())
    mlflow.log_metric("overall_accuracy", overall_accuracy)

    # Calculate accuracy per tag (expanding tags since case_tags is a list)
    df_exploded = df_valid.explode("tags")
    accuracy_per_tag = df_exploded.groupby("tags")["evaluator_result"].mean()
    for tag, accuracy in accuracy_per_tag.items():
        if pd.notna(tag):  # pyright: ignore[reportUnknownMemberType]
            mlflow.log_metric(f"accuracy_tag_{tag}", float(accuracy))

    # for each row, log the feedback to mlflow:
    for _, row in eval_result_df.iterrows():
        row_data: Any = row
        trace_id: str = str(row_data["trace_id"])
        feedback = Feedback(
            name=str(row_data["evaluator_name"]),
            value=row_data["evaluator_result"],
            source=AssessmentSource(source_type=str(row_data["source_type"]), source_id=str(row_data["source_id"])),
            rationale=str(row_data["evaluator_reason"]),
        )
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    # for each reportcase:
    for input_key, reportcase in input_key_report_case_map.items():
        trace_id = str(eval_df.loc[eval_df["input_key"] == input_key, "trace_id"].iloc[0])
        assert isinstance(reportcase.metadata, TestCaseMetadata), "ReportCase metadata is not of type TestCaseMetadata."
        for key, value in reportcase.metadata.attributes.items():
            mlflow.set_trace_tag(trace_id, key, str(value))
        for tag in reportcase.metadata.tags:
            mlflow.set_trace_tag(trace_id, f"tag_{tag}", "true")


async def evaluate_testset_with_mlflow(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Evaluate a testset with comprehensive MLflow logging and tracking.

    This function orchestrates the complete evaluation workflow:

    1. Sets up MLflow experiment and starts a run
    2. Wraps the task with MLflow tracing
    3. Evaluates all test cases using the provided task
    4. Cleans up LLMJudge traces (which clutter the UI)
    5. Maps traces to evaluation results
    6. Logs metrics, parameters, and assessments to MLflow
    7. Tags traces with case metadata for filtering and analysis

    The function automatically:

    - Logs overall accuracy
    - Logs accuracy per tag for granular analysis
    - Attaches feedback/assessments to each trace
    - Preserves trace IDs for later inspection
    - Logs model parameters for reproducibility

    Args:
        testset: The dataset to evaluate, created via
            [`load_testset`][ragpill.csv.testset.load_testset]
            or constructed manually using [`Case`](https://ai.pydantic.dev/api/pydantic_evals/dataset/#pydantic_evals.dataset.Case) objects
        task: The task to evaluate - can be either synchronous or asynchronous callable.
            Should accept inputs of type `InputsT` and return outputs of type `OutputT`.
            Example: `async def my_agent(question: str) -> str: ...`
        mlflow_settings: MLflow configuration settings. If None, loads from environment variables:
            - `EVAL_MLFLOW_TRACKING_URI`: MLflow tracking server URI
            - `EVAL_MLFLOW_EXPERIMENT_NAME`: Experiment name for grouping runs
            - `EVAL_MLFLOW_TRACKING_USERNAME`: Authentication username (if needed)
            - `EVAL_MLFLOW_TRACKING_PASSWORD`: Authentication password (if needed)
        model_params: Optional dictionary of model/system parameters to log for reproducibility.
            Examples: `{"system_prompt": "...", "model": "gpt-4o", "temperature": "0.7",
            "retrieval_k": "5", "rerank_model": "..."}`

    Returns:
        pandas.DataFrame: Evaluation results with columns:
            - `inputs`: Test case input
            - `output`: Task output
            - `evaluator_result`: Boolean pass/fail result
            - `evaluator_data`: Evaluator-specific data (e.g., rubric for LLMJudge)
            - `evaluator_reason`: Explanation for the result
            - `expected`: Whether pass was expected
            - `attributes`: JSON-encoded custom attributes
            - `tags`: Set of tags for categorization
            - `task_duration`: Time taken for task execution
            - `evaluator_name`: Name of the evaluator
            - `case_name`: Name of the test case
            - `case_id`: Unique identifier for the case
            - `source_type`: "LLM_JUDGE" or "CODE"
            - `source_id`: Evaluator class name
            - `input_key`: Hash of the input
            - `trace_id`: MLflow trace ID for inspection

    Example:
        ```python
        import mlflow
        from ragpill.csv.testset import load_testset, default_evaluator_classes
        from ragpill.mlflow_helper import evaluate_testset_with_mlflow
        from ragpill.settings import MLFlowSettings

        # Load test dataset
        testset = load_testset(
            csv_path="testset.csv",
            evaluator_classes=default_evaluator_classes,
        )

        # Define your task
        async def my_agent(question: str) -> str:
            # Your agent logic here
            return f"Answer to: {question}"

        # Run evaluation with MLflow tracking
        results_df = evaluate_testset_with_mlflow(
            testset=testset,
            task=my_agent,
            model_params={
                "model": "gpt-4o-mini",
                "temperature": "0.7",
                "system_prompt": "You are a helpful assistant",
            }
        )

        # Analyze results
        print(f"Overall accuracy: {results_df['evaluator_result'].mean():.2%}")
        ```

    Note:
        This function will start and end an MLflow run. Make sure MLflow tracking
        is properly configured before calling this function.

    See Also:
        [`load_testset`][ragpill.csv.testset.load_testset]:
            Create test datasets from CSV files
        [`MLFlowSettings`][ragpill.settings.MLFlowSettings]:
            MLflow configuration settings
    """
    mlflow_settings = mlflow_settings or MLFlowSettings()  # pyright: ignore[reportCallIssue]
    _setup_mlflow_experiment(mlflow_settings)
    _fix_evaluator_global_flag(testset)
    testsetresults = await testset.evaluate(_mlflow_runnable_wrapper(task))
    experiment, latest_run_id = _delete_llm_judge_traces(mlflow_settings)
    input_key_trace_map = _get_input_key_trace_id_map(experiment, latest_run_id)
    input_key_report_case_map = _get_input_key_report_case_map(testsetresults, testset)
    eval_metadata_map = _get_evaluation_id_eval_metadata_map(testset)
    assert set(input_key_trace_map.keys()) == set(
        input_key_report_case_map.keys()
    ), "Input keys in traces and testsetresults do not match."
    eval_result_df = _create_evaluation_dataframe(
        input_key_trace_map,
        input_key_report_case_map,
        eval_metadata_map,
    )
    _upload_mlflow(eval_result_df, input_key_report_case_map, model_params)

    mlflow.end_run()
    return eval_result_df


def evaluate_testset_with_mlflow_sync(
    testset: Dataset[Any, Any, CaseMetadataT],
    task: TaskType,
    mlflow_settings: MLFlowSettings | None = None,
    model_params: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Synchronous wrapper around [`evaluate_testset_with_mlflow`][ragpill.mlflow_helper.evaluate_testset_with_mlflow].

    Prefer the async version when possible. Use this wrapper when you cannot use `await` —
    for example in plain scripts, CLI tools, or synchronous test suites.

    Internally, this runs the async function via `asyncio.run()` inside a fresh thread from a
    `ThreadPoolExecutor`. That thread has no running event loop, so `asyncio.run()` always
    succeeds — even when the *caller* is already inside a running event loop (e.g. Jupyter,
    FastAPI, or an `asyncio`-based test runner).

    Args:
        testset: The dataset to evaluate.
        task: The task to evaluate — sync or async callable.
        mlflow_settings: MLflow configuration. If None, loaded from environment variables.
        model_params: Optional model/system parameters to log for reproducibility.

    Returns:
        pandas.DataFrame: Same evaluation results as the async version.

    See Also:
        [`evaluate_testset_with_mlflow`][ragpill.mlflow_helper.evaluate_testset_with_mlflow]:
            The async version of this function.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            asyncio.run,
            evaluate_testset_with_mlflow(testset, task, mlflow_settings, model_params),
        )
        return future.result()
