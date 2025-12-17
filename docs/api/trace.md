# Trace Model

Vendor-neutral trace model consumed by the renderer and span-based evaluators.
A captured backend trace (today `mlflow.entities.Trace`) is converted to a
[`Trace`][ragpill.trace.Trace] at the capture boundary via
[`from_mlflow_trace`][ragpill.trace.from_mlflow_trace]; nothing downstream sees
the backend's native shape. Dialect adapters (Option C) convert normalised
OTLP-JSON span dicts into [`Span`][ragpill.trace.Span] objects.

## Trace

::: ragpill.trace.Trace
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Span

::: ragpill.trace.Span
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## SpanKind

::: ragpill.trace.SpanKind
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Message

::: ragpill.trace.Message
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Document

::: ragpill.trace.Document
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage

::: ragpill.trace.Usage
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## parse_otel

::: ragpill.trace.parse_otel
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## detect_dialect

::: ragpill.trace.detect_dialect
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## from_mlflow_trace

::: ragpill.trace.from_mlflow_trace
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## filter_to_subtree

::: ragpill.trace.filter_to_subtree
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## trace_to_dict

::: ragpill.trace.trace_to_dict
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## trace_from_dict

::: ragpill.trace.trace_from_dict
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## adapters.SpanAdapter

::: ragpill.trace.adapters.SpanAdapter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## adapters.MLflowAdapter

::: ragpill.trace.adapters.MLflowAdapter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## adapters.GenAIAdapter

::: ragpill.trace.adapters.gen_ai.GenAIAdapter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## adapters.OpenInferenceAdapter

::: ragpill.trace.adapters.openinference.OpenInferenceAdapter
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
