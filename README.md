# RAGPill

Opinionated but customizable wrappers around [pydantic-ai's evaluation framework](https://ai.pydantic.dev/evals/) using [MLflow tracing](https://mlflow.org/docs/latest/genai/tracing/) for testing LLM systems. Follow best practices while maintaining flexibility.

## What is RAGPill?

RAGPill helps you:

- **Create test datasets from CSV files** - Easy collaboration with domain experts
- **Define custom evaluators** - Add domain-specific knowledge to evaluations
- **Track results in MLflow** - Full experiment tracking and tracing
- **Follow best practices** - Opinionated design guides you to robust testing

It specializes in "offline" evaluation of LLM-based systems, meaning it's supposed to be part of your CI/CD pipeline or run as scheduled tests, not real-time monitoring.

## Demo!

TODO: this should be a video demo
Prerequisites:
- [ragpill installed](docs/getting-started/installation.md)
- MLflow tracking server running (local or remote) with tracing enabled.
Locally: `mlflow server --backend-store-uri sqlite:///mlflow.db` or if remote, then configure the env vars properly.

```python


```
Produces the following mlflow views:

### Metrics View
[![ragpill Demo](https://img.youtube.com/vi/your-video-id/0.jpg)](https://www.youtube.com/watch?v=your-video-id)

### Traces Views

### Artifacts View
[![ragpill Demo Artifacts](https://img.youtube.com/vi/your-video-id/0.jpg)](https://www.youtube.com/watch?v=your-video-id)

### Comparing runs

### What's wrong with other frameworks?

- The [pydantic-ai's evaluation framework](https://ai.pydantic.dev/evals/) only integrates with cloud-based logfire and it sucks for mlflow tracing (if you go the hassle and use mlflow opentelemetry endpoint as logfire backend, a lot of mlflow features are lost in translation). However, we like the core concepts and type-safety of pydantic-ai evals a lot, so we build on top of it.
- additionally, it's not straightforward to test for example, if a regex pattern is found in retrieved sources or document metadata. Which is common enough in retrieval-augmented generation (RAG) systems.
- [Langsmith evaluation](https://docs.langchain.com/langsmith/evaluation) doesn't support multiple tests per dataset item, nor custom evaluators easily. Also no mlflow integration.



## Core Philosophy

Here we focus a lot on the [LLM Judge evaluator](docs/api/evaluators.md#llmjudge), although it's the last evaluator you should use - prefer deterministic evaluators (regex, exact match) whenever possible. 
However, for deterministic tests, there's already a lot of tooling available, like pytest for example (yes, we like the 'code-first' approach).

### Expert-Defined Attributes

LLM judges usually lack context awareness to judge which discrepancies between chatbot answers and expected answers are relevant - especially in specialized fields like law, engineering, and science where words have precise definitions.

**Domain experts should define specific attributes and criteria for evaluation.**

### Binary Evaluations

We use **boolean pass/fail values only**, not scoring scales (1-10), because:

- Scales are arbitrary and often decided by LLMs
- Binary decisions are more stable and reproducible (although LLMs of course remain probabilistic)
- Easier to track and reason about over time

### Tags and Attributes for Organization

Evaluators can have:

- **Tags**: Categorical labels for filtering (e.g., `retrieval`, `time-aware-rag`, `basic_logic`)
- **Attributes**: Key-value metadata for categorization (e.g., `importance: high`, `scope: Phase1`)

Metrics are automatically calculated per tag and attribute.


## Quick Navigation

### Getting Started:

- [Installation](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)

### Evaluators:

## Key Concepts

As this library is built on pydantic-ai evals, please have a look [here](https://ai.pydantic.dev/evals/core-concepts/)


### Key Components

- **Dataset**: From pydantic-ai, contains test cases with inputs, evaluators, and metadata
- **Evaluators**: Check if outputs meet criteria (LLMJudge, regex matchers, custom evaluators)
- **MLflow Integration**: Wraps execution, traces runs, evaluates outputs, uploads results

## Features

- **Great MLflow Integration**: Traces your agent/function execution to MLflow with evaluations in the native format
- **CSV/Excel Adapter**: Load test cases from CSV files with evaluator configurations
- **Flexible Evaluators**: Built-in LLM judges, regex matchers, and easy custom evaluator creation
- **Metrics per Tags/Attributes**: Automatic metric calculation for each tag and attribute combination
- **Type Safety**: Built on pydantic-ai with full type safety throughout

## [Built-in Evaluators](docs/api/evaluators.md)

- [**LLMJudge**](docs/api/evaluators.md#llmjudge): Uses an LLM to judge correctness based on a rubric
- [**RegexInSourcesEvaluator**](docs/api/evaluators.md#regexinsourcesevaluator): Checks if regex patterns appear in retrieved sources
- [**RegexInDocumentMetadataEvaluator**](docs/api/evaluators.md#regexindocumentmetadataevaluator): Checks regex in document metadata
- [**Custom Evaluators**](docs/guide/evaluators.md#creating-custom-evaluators): Inherit from `BaseEvaluator` and implement your logic

## Best Practices

!!! tip "TDD Mindset"
    Begin with defining a Test-Set with potential users before even starting to develop the solution. This enables clear expectation management and progress tracking.

!!! tip "Create Multiple Testsets"
    It might make sense for you to have some core tests that run relatively quickly and inexpensive - use these for development. Before deploying to prod, you can run an exhaustive dataset that is integrated in your CI/CD.

!!! tip "Separate Evaluation Experiments"
    Create dedicated MLflow experiments for evaluations. Don't mix evaluation traces with production traces.

!!! tip "Use Domain Experts"
    Have domain experts define evaluation criteria rather than relying solely on generic LLM judges.

!!! tip "Version Your Tests"
    Keep test datasets in version control alongside your code.

## Documentation

Full documentation is available at [your-docs-url] including:

- **Installation Guide**: Setup instructions
- **Quickstart Tutorial**: Run your first evaluation
- **CSV Adapter Guide**: Learn the CSV format and column meanings
- **Evaluators Guide**: Create custom evaluators
- **MLflow Integration**: Advanced MLflow usage
- **API Reference**: Complete API documentation

## Roadmap

- [x] Adapter for testset from CSV
- [x] Documentation via mkdocs
- [x] Evaluators for sources and regex
- [ ] Adapter for task from CSV (upload to mlflow)
- [ ] Create demo video
- [ ] CI/CD (tests, build package, publish docs)
- [ ] Global evaluators from CSV (empty input)
- [ ] Track git-commit hash
- [ ] Tests with test-docker containers
- [ ] Dependency injection for llm, input_to_key functions
- [ ] pytest integration (mandatory throws errors, non-mandatory logs warnings)


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/development/contributing.md) for guidelines.
