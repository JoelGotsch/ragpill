# Evaluators

This module provides pre-built evaluators for common evaluation tasks.

As the task can have arbitrary input and output types these evaluators in general coerce those to strings and use string-based evaluation methods (LLM judges, regex checks, etc).

It is, however, pretty easy to create your own custom evaluators by inheriting from `BaseEvaluator` and implementing the `run()` method. See [Create custom evaluators](../how-to/custom-type-evaluator.ipynb) for a tutorial on how to do this.

## LLMJudge

::: ragpill.evaluators.LLMJudge
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## RegexInSourcesEvaluator

::: ragpill.evaluators.RegexInSourcesEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## RegexInDocumentMetadataEvaluator

::: ragpill.evaluators.RegexInDocumentMetadataEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## RegexInOutputEvaluator

::: ragpill.evaluators.RegexInOutputEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## LiteralQuotationTest

::: ragpill.evaluators.LiteralQuotationTest
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## HasQuotesEvaluator

::: ragpill.evaluators.HasQuotesEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3


## Base Evaluators

These are Evaluators that are useful to inherit from. See [Create custom evaluators](../guide/evaluators.md#creating-custom-evaluators)

### WrappedPydanticEvaluator

::: ragpill.evaluators.WrappedPydanticEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SpanBaseEvaluator

::: ragpill.evaluators.SpanBaseEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### SourcesBaseEvaluator

::: ragpill.evaluators.SourcesBaseEvaluator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## See Also

- [Create custom evaluators](../guide/evaluators.md#creating-custom-evaluators)
