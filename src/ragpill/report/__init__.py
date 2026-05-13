"""LLM-readable markdown views of ragpill outputs.

See ``designs/llm-readable-outputs-and-mcp.md`` for the design rationale.
"""

from ragpill.report.exploration import render_dataset_run_as_exploration
from ragpill.report.triage import render_evaluation_output_as_triage

__all__ = [
    "render_dataset_run_as_exploration",
    "render_evaluation_output_as_triage",
]
