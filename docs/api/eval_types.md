# Evaluation Primitives

Local evaluation primitives that replaced the previously-required
`pydantic_evals` types. These are plain dataclasses — cheap to construct,
serializable, and decoupled from upstream framework changes.

## EvaluationReason

::: ragpill.eval_types.EvaluationReason
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## EvaluationResult

::: ragpill.eval_types.EvaluationResult
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## EvaluatorSource

::: ragpill.eval_types.EvaluatorSource
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## EvaluatorContext

::: ragpill.eval_types.EvaluatorContext
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Case

::: ragpill.eval_types.Case
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Dataset

::: ragpill.eval_types.Dataset
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [Base Classes](base.md) — `BaseEvaluator` uses these types.
- [Evaluators](evaluators.md) — all evaluators build on these primitives.
