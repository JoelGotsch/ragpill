"""End-to-end pipeline integration tests: execute → evaluate → upload.

Two rich scenarios against a real local SQLite MLflow store (no external
server required):

1. ``test_pipeline_edge_cases`` — one dataset packed with edge cases (task
   errors, unicode + regex metacharacters, empty outputs, flaky repeats with
   thresholds, an evaluator that raises, a global evaluator) driven through
   execute + evaluate + both JSON round-trips.
2. ``test_pipeline_all_bells_end_to_end`` — an intense, diverse run with all
   the bells: retriever spans with documents, span-based + output + LLM-judge
   evaluators (TestModel), tags/attributes, repeats, a real tracking-server
   upload, and server-side assertions on metrics, assessments, tags, and
   artifacts.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import mlflow
import pytest
from pydantic_ai.models.test import TestModel

from ragpill.base import BaseEvaluator, EvaluatorMetadata, TestCaseMetadata
from ragpill.eval_types import Case, Dataset, EvaluationReason, EvaluatorContext
from ragpill.evaluation import evaluate_results
from ragpill.evaluators import (
    LLMJudge,
    RegexInDocumentMetadataEvaluator,
    RegexInOutputEvaluator,
    RegexInSourcesEvaluator,
)
from ragpill.execution import DatasetRunOutput, execute_dataset
from ragpill.settings import TrackingSettings
from ragpill.types import EvaluationOutput
from ragpill.upload import upload_results

pytestmark = [
    pytest.mark.anyio(backends=["asyncio"]),
    pytest.mark.parametrize("anyio_backend", ["asyncio"]),
]


@dataclass(kw_only=True, repr=False)
class _ExplodingEvaluator(BaseEvaluator):
    """Always raises — exercises the evaluator_failures path."""

    async def run(self, ctx: EvaluatorContext[Any, Any, EvaluatorMetadata]) -> EvaluationReason:
        raise RuntimeError("evaluator exploded on purpose")


# ---------------------------------------------------------------------------
# Scenario 1 — edge cases
# ---------------------------------------------------------------------------


async def test_pipeline_edge_cases():
    """One dataset, many edges: unicode + regex metachars, task errors, empty
    outputs, flaky repeats vs threshold, a raising evaluator, a global
    evaluator — through execute, evaluate, and both JSON round-trips."""

    async def task(q: str) -> str:
        if q == "explode":
            raise ValueError("task blew up (deliberately)")
        if q == "empty":
            return ""
        return f"answer: {q}"

    # A stateful factory: the flaky case's runs alternate good/bad output.
    flaky_counter = {"n": 0}

    def task_factory():
        async def flaky(q: str) -> str:
            if q == "flaky?":
                flaky_counter["n"] += 1
                return "good" if flaky_counter["n"] % 3 else "bad"
            return await task(q)

        return flaky

    unicode_case = Case(
        inputs="was kostet f(x)=x² in München?",
        metadata=TestCaseMetadata(tags={"unicode"}),
        evaluators=[RegexInOutputEvaluator(pattern="München", tags={"unicode"})],
    )
    error_case = Case(
        inputs="explode",
        metadata=TestCaseMetadata(tags={"errors"}),
        evaluators=[RegexInOutputEvaluator(pattern="anything")],
    )
    empty_output_case = Case(
        inputs="empty",
        metadata=TestCaseMetadata(),
        evaluators=[RegexInOutputEvaluator(pattern="answer")],  # "" does not match
    )
    # Runs 1..3: good, good, bad → pass rate 2/3 clears the 0.5 threshold.
    flaky_case = Case(
        inputs="flaky?",
        metadata=TestCaseMetadata(repeat=3, threshold=0.5, tags={"flaky"}),
        evaluators=[RegexInOutputEvaluator(pattern="^good$")],
    )
    exploding_case = Case(
        inputs="normal",
        metadata=TestCaseMetadata(),
        evaluators=[_ExplodingEvaluator(), RegexInOutputEvaluator(pattern="answer")],
    )
    testset = Dataset[str, str, TestCaseMetadata](
        cases=[unicode_case, error_case, empty_output_case, flaky_case, exploding_case],
        # Matches every output this dataset produces (incl. the empty string).
        evaluators=[RegexInOutputEvaluator(pattern="answer|good|bad|^$", is_global=True, tags={"global"})],
    )

    run_output = await execute_dataset(testset, task_factory=task_factory, capture_traces=True)

    # --- execution edges ---------------------------------------------------
    assert len(run_output.cases) == 5
    error_run = run_output.cases[1].task_runs[0]
    assert error_run.error is not None and "deliberately" in error_run.error
    assert error_run.output is None
    assert run_output.cases[2].task_runs[0].output == ""
    assert len(run_output.cases[3].task_runs) == 3
    # Every non-erroring run captured a trace scoped to its own span.
    ok_run = run_output.cases[0].task_runs[0]
    assert ok_run.trace is not None and ok_run.run_span_id

    # Run-JSON round-trip preserves everything, including the error string.
    restored = DatasetRunOutput.from_json(run_output.to_json())
    assert restored.cases[1].task_runs[0].error == error_run.error
    assert restored.cases[3].task_runs[2].output == "bad"

    evaluation = await evaluate_results(run_output, testset)

    # --- evaluation edges --------------------------------------------------
    by_name = {cr.case_name: cr for cr in evaluation.case_results}

    unicode_cr = by_name["was kostet f(x)=x² in München?"]
    assert unicode_cr.run_results[0].all_passed

    # Task error short-circuits every evaluator (incl. the global one) to False.
    error_cr = by_name["explode"]
    assert not error_cr.aggregated.passed
    assert all(r.value is False for r in error_cr.run_results[0].assertions.values())
    assert "Task execution failed" in next(iter(error_cr.run_results[0].assertions.values())).reason

    # Empty output: the pattern evaluator fails, the permissive global one passes.
    empty_cr = by_name["empty"]
    empty_assertions = empty_cr.run_results[0].assertions
    assert empty_assertions["RegexInOutputEvaluator"].value is False

    # 2/3 runs pass ≥ 0.5 threshold → case passes with pass_rate 2/3.
    flaky_cr = by_name["flaky?"]
    assert flaky_cr.aggregated.passed
    assert flaky_cr.aggregated.pass_rate == pytest.approx(2 / 3)
    assert flaky_cr.aggregated.per_evaluator_pass_rates["RegexInOutputEvaluator"] == pytest.approx(2 / 3)

    # A raising evaluator lands in evaluator_failures without flipping the verdict.
    exploding_cr = by_name["normal"]
    failures = exploding_cr.run_results[0].evaluator_failures
    assert len(failures) == 1 and "exploded on purpose" in failures[0].error_message
    assert exploding_cr.run_results[0].all_passed  # remaining assertions all passed

    # Global evaluator ran on every case; per-tag accuracy sees the case tags.
    assert all(
        "RegexInOutputEvaluator" in rr.assertions or rr.error for cr in evaluation.case_results for rr in cr.run_results
    )
    tag_accuracy = evaluation.per_tag_accuracy()
    assert tag_accuracy["unicode"] == pytest.approx(1.0)
    assert 0.0 < tag_accuracy["flaky"] < 1.0

    # EvaluationOutput JSON round-trip preserves results and the dataset_run.
    restored_eval = EvaluationOutput.from_json(evaluation.to_json())
    assert len(restored_eval.case_results) == 5
    assert restored_eval.case_results[3].aggregated.pass_rate == pytest.approx(2 / 3)
    assert restored_eval.dataset_run is not None
    assert restored_eval.case_results[4].run_results[0].evaluator_failures[0].error_message == failures[0].error_message

    # The triage rendering handles the whole mix without blowing up.
    triage = evaluation.to_llm_text()
    assert "explode" in triage and len(triage) > 200


# ---------------------------------------------------------------------------
# Scenario 2 — all bells: retriever spans, judge, tags, upload, server checks
# ---------------------------------------------------------------------------


async def test_pipeline_all_bells_end_to_end():
    """Diverse, intense usage of the full pipeline against a real local
    tracking store: retriever spans feeding span-based evaluators, an LLM
    judge on TestModel, tags/attributes/repeats, then a real upload with
    server-side assertions on metrics, assessments, trace tags, and artifacts."""
    server_dir = tempfile.mkdtemp(prefix="ragpill_server_")
    server_uri = f"sqlite:///{os.path.join(server_dir, 'server.db')}"
    experiment = "ragpill_pipeline_bells"
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(server_uri)
        mlflow.create_experiment(experiment, artifact_location=f"file://{server_dir}/artifacts")

        docs = [
            {"page_content": "The capital of France is Paris.", "metadata": {"source": "geo.txt"}},
            {"page_content": "Paris hosts the Louvre museum.", "metadata": {"source": "culture.txt"}},
        ]

        async def rag_task(q: str) -> str:
            # Emit a real retriever span so span-based evaluators see documents.
            with mlflow.start_span(name="retrieve", span_type="RETRIEVER") as span:
                span.set_inputs(q)
                span.set_outputs(docs)
            return f'According to geo.txt: "The capital of France is Paris." ({q})'

        judge = LLMJudge(rubric="Answer must name a city.", model=TestModel(), tags={"judge"})
        geo_case = Case(
            inputs="capital of France?",
            metadata=TestCaseMetadata(repeat=2, tags={"geo"}, attributes={"difficulty": "easy"}),
            evaluators=[
                RegexInOutputEvaluator(pattern="Paris"),
                RegexInSourcesEvaluator.from_csv_line(expected=True, tags={"sources"}, check="capital of France"),
                RegexInDocumentMetadataEvaluator.from_csv_line(
                    expected=True, tags={"sources"}, check='{"pattern": "geo.txt", "key": "source"}'
                ),
            ],
        )
        culture_case = Case(
            inputs="museum in Paris?",
            metadata=TestCaseMetadata(tags={"culture"}, attributes={"difficulty": "hard"}),
            evaluators=[RegexInOutputEvaluator(pattern="Paris")],
        )
        testset = Dataset[str, str, TestCaseMetadata](cases=[geo_case, culture_case], evaluators=[judge])

        settings = TrackingSettings(
            tracking_uri=server_uri,
            experiment_name=experiment,
        )
        run_output = await execute_dataset(
            testset, task=rag_task, capture_traces=True, tracking_uri=server_uri, settings=settings
        )

        # Execution captured a run + per-repeat traces with the retriever span inside.
        assert run_output.run_id and run_output.experiment_id
        assert len(run_output.cases[0].task_runs) == 2
        run0 = run_output.cases[0].task_runs[0]
        assert run0.trace is not None
        assert any(s.name == "retrieve" for s in run0.trace.spans)

        # Evaluate while pointed at the server (as the judge writes spans).
        mlflow.set_tracking_uri(server_uri)
        evaluation = await evaluate_results(run_output, testset, settings)

        geo_cr, culture_cr = evaluation.case_results
        # Span-based evaluators found the retriever documents on every repeat.
        for rr in geo_cr.run_results:
            assert rr.assertions["RegexInOutputEvaluator"].value is True
            assert rr.assertions["RegexInSourcesEvaluator"].value is True
            assert rr.assertions["RegexInDocumentMetadataEvaluator"].value is True
            assert "Rubric:" in rr.assertions["LLMJudge"].reason
        assert geo_cr.aggregated.passed is bool(geo_cr.aggregated.pass_rate >= geo_cr.aggregated.threshold)
        assert culture_cr.run_results[0].assertions["RegexInOutputEvaluator"].value is True

        # Per-attribute accuracy discovered the difficulty attribute.
        assert set(evaluation.per_attribute_accuracy("difficulty")) == {"easy", "hard"}

        upload_results(evaluation, settings=settings, model_params={"model": "test"}, upload_traces=True)

        # --- server-side assertions -----------------------------------------
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=server_uri)
        run = client.get_run(run_output.run_id)
        metrics = run.data.metrics
        assert metrics["overall_accuracy"] > 0.0
        assert "accuracy_tag_geo" in metrics and "accuracy_tag_judge" in metrics
        assert "accuracy_attr_difficulty_easy" in metrics
        assert run.data.params.get("model") == "test"

        artifact_paths = {a.path for a in client.list_artifacts(run_output.run_id)}
        assert "evaluation_results.json" in artifact_paths  # the runs table
        assert "ragpill_traces" in artifact_paths  # upload_traces=True payload

        # Per-run assessments and case tags landed on the per-repeat traces.
        for rr in geo_cr.run_results:
            trace_info = client.get_trace(rr.trace_id).info
            assessment_names = {a.name for a in trace_info.assessments}
            assert f"run-{rr.run_index}_RegexInOutputEvaluator" in assessment_names
            assert "agg_RegexInOutputEvaluator" in assessment_names  # repeat=2 → aggregates too
            assert trace_info.tags.get("tag_geo") == "true"
            assert trace_info.tags.get("difficulty") == "easy"
            # Session grouping + case-level metadata carried on each repeat.
            assert trace_info.trace_metadata.get("mlflow.trace.session") == geo_cr.base_input_key
            assert trace_info.trace_metadata.get("ragpill.case_name") == "capital of France?"

        # The exploration rendering reflects the (renamed) run/experiment ids.
        exploration = run_output.to_llm_text()
        assert run_output.run_id in exploration
    finally:
        mlflow.set_tracking_uri(previous_uri)
        shutil.rmtree(server_dir, ignore_errors=True)
