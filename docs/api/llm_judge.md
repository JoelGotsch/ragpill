# LLM Judge

Low-level LLM-as-a-judge helpers used by
[`LLMJudge`](evaluators.md#llmjudge). These wrap a `pydantic_ai.Agent` and
return a structured `GradingOutput`.

## GradingOutput

::: ragpill.llm_judge.GradingOutput
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## judge_output

::: ragpill.llm_judge.judge_output
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## judge_input_output

::: ragpill.llm_judge.judge_input_output
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## See Also

- [`LLMJudge`](evaluators.md#llmjudge) — the high-level evaluator that wraps
  these functions.
