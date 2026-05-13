# Reporting (LLM-Readable Views)

Markdown renderers that convert ragpill outputs into a form an LLM (or a human
in a chat session) can read directly. See the
[LLM Reports Guide](../guide/llm-reports.md) for when to use which view.

## render_evaluation_output_as_triage

::: ragpill.report.triage.render_evaluation_output_as_triage
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## render_dataset_run_as_exploration

::: ragpill.report.exploration.render_dataset_run_as_exploration
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [LLM Reports Guide](../guide/llm-reports.md) — when to use triage vs exploration.
- [Result Types](types.md) — the `EvaluationOutput.to_llm_text` /
  `EvaluationOutput.to_json` shortcuts that delegate to these renderers.
- [Execution Layer](execution.md) — `DatasetRunOutput.to_llm_text` likewise.
