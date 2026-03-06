# Overview

This document provides an overview of the `ragpill` framework and its major components.

A Dataset is a collection of Test-[Case](https://ai.pydantic.dev/evals/core-concepts/#case)s, each of which represent an input (usually a question to the LLM).
Each Test-[Case](https://ai.pydantic.dev/evals/core-concepts/#case) can contain multiple Evaluators, each of which represent a test for a different aspect of the LLM output (or artifact for generating that output).

Each input is first processed by a Task (usually a llm-agent), which generates an output and (if configured correctly) mlflow traces.
The evaluators then can compare the output and traces to the specified criteria, like if a certain fact is part of the answer, a certain document was retrieved in the process, etc.
Each evaluator for each of the Test-[Case](https://ai.pydantic.dev/evals/core-concepts/#case)s, a EvaluationReason is generated.

!!! note "Error Handling During Task Execution"
    If an error occurs during task execution (e.g., LLM timeouts, parsing errors, or other runtime exceptions), all associated test cases for that input will evaluate to `False`. This ensures that execution failures are properly reflected in the evaluation results.


