# Test Set Guide

This guide will show you how to create test sets for evaluating your agents/functions using ragpill. We will cover both defining test sets in code and loading them from CSV files, but we focus on the ideas behind test sets and best practices.

## Key principles

When creating test sets, keep the following principles in mind:
- **Range of scenarios**: Cover common, edge, and failure cases
- **Create realistic inputs**: Use inputs that reflect real-world usage, not idealized question formulations
- **Deterministic first**: Start with deterministic checks (e.g., regex match) before moving to probabilistic ones (e.g., LLM judge)
- **LLMJudge**: Think of it like an autistic kid that follows instructions literally. Be explicit in your rubric.


## Repeat and Threshold

Each test case can specify how many times to run and what pass fraction is required:

- **`repeat`**: Number of times to execute the task for this case (default: 1, inherited from `MLFlowSettings.ragpill_repeat`).
- **`threshold`**: Minimum fraction of runs that must pass (default: 1.0, inherited from `MLFlowSettings.ragpill_threshold`).

Set these on `TestCaseMetadata` when building test cases programmatically:

```python
case = Case(
    inputs="What is the capital of France?",
    metadata=TestCaseMetadata(repeat=3, threshold=0.8),
    evaluators=[...],
)
```

Or add `repeat` and `threshold` columns in your CSV files. See the [Repeated Runs Guide](repeated-runs.md) for full details.

## Best Practices

!!! tip "TDD Mindset"
    Begin with defining a Test-Set with potential users before even starting to develop the solution. This enables clear expectation management and progress tracking.
