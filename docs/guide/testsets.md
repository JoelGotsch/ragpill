# Test Set Guide

This guide will show you how to create test sets for evaluating your agents/functions using ragpill. We will cover both defining test sets in code and loading them from CSV files, but we focus on the ideas behind test sets and best practices.

## Key principles

When creating test sets, keep the following principles in mind:
- **Range of scenarios**: Cover common, edge, and failure cases
- **Create realistic inputs**: Use inputs that reflect real-world usage, not idealized question formulations
- **Deterministic first**: Start with deterministic checks (e.g., regex match) before moving to probabilistic ones (e.g., LLM judge)
- **LLMJudge**: Think of it like an autistic kid that follows instructions literally. Be explicit in your rubric.


## Best Practices

!!! tip "TDD Mindset"
    Begin with defining a Test-Set with potential users before even starting to develop the solution. This enables clear expectation management and progress tracking.
