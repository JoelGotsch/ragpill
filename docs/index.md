# RAGPill Documentation

RAGPill is a library for **granular, expert-driven evaluation of LLM and RAG
applications**. You curate test cases (often from a CSV your domain experts
maintain), attach binary pass/fail evaluators — regex checks, quote-verification,
source checks, and an LLM judge as a last resort — and run your pipeline against
them with per-tag metrics, repeated runs for statistical confidence, and
pluggable tracking backends (MLflow by default; Langfuse and Arize Phoenix are
also supported).

## Getting Started

- [Installation](getting-started/installation.md) — including the backend extras
  (`ragpill[mlflow]` / `[langfuse]` / `[phoenix]`).
- [Quick Start Guide](getting-started/quickstart.md) — first evaluation in a few
  minutes, zero server required.

## Guides

- [Overview](guide/overview.md)
- [Layered Architecture](guide/layered-architecture.md) — execute → evaluate →
  upload as three independent layers (run once, evaluate many; CI without a server).
- [Test Sets](guide/testsets.md) and the [CSV Adapter](guide/csv-adapter.md)
- [Evaluators](guide/evaluators.md)
- [Repeated Runs](guide/repeated-runs.md)
- [LLM-readable Reports](guide/llm-reports.md)

## How-To

- [Write a custom evaluator](how-to/custom-evaluator.md)
- [Evaluate historical outputs](how-to/evaluate-historical-outputs.md)
- [Use a task factory for stateful tasks](how-to/task-factory.md)

## Reference

- [API Reference](api/base.md)
- [Architecture Decision Records](adr/index.md)

## Tutorial

First time here? Work through the end-to-end
[tutorial](tutorials/full.md).
