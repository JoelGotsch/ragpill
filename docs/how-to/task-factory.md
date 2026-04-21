# Task Factory

This guide explains when and how to use `task_factory` instead of `task` when running repeated evaluations.

## The Problem

When using `repeat > 1` with a stateful task (e.g., an agent that accumulates message history), the same task instance is reused across runs. State from run 0 leaks into run 1, contaminating results:

```python
class MyAgent:
    def __init__(self):
        self.history = []

    async def __call__(self, question: str) -> str:
        self.history.append(question)
        # Bug: history grows across runs!
        return f"Answer (seen {len(self.history)} questions)"

agent = MyAgent()

# With task=agent and repeat=3:
# run-0: history = ["What is X?"] → "seen 1 questions"
# run-1: history = ["What is X?", "What is X?"] → "seen 2 questions"  # contaminated!
# run-2: history = ["What is X?", "What is X?", "What is X?"] → "seen 3 questions"
```

## The Solution

Use `task_factory` to create a fresh task instance for each run:

```python
def create_agent():
    return MyAgent()  # Fresh instance with empty history

result = await evaluate_testset_with_mlflow(
    testset=testset,
    task_factory=create_agent,  # Called once per run
)

# With task_factory and repeat=3:
# run-0: fresh agent, history = ["What is X?"] → "seen 1 questions"
# run-1: fresh agent, history = ["What is X?"] → "seen 1 questions"  # clean!
# run-2: fresh agent, history = ["What is X?"] → "seen 1 questions"  # clean!
```

## pydantic-ai Agent Example

A common pattern with `pydantic-ai` agents that use `message_history`:

```python
from pydantic_ai import Agent

def create_agent():
    agent = Agent(
        "openai:gpt-4o",
        system_prompt="You are a helpful assistant.",
    )

    async def task(question: str) -> str:
        result = await agent.run(question)
        return result.output

    return task

result = await evaluate_testset_with_mlflow(
    testset=testset,
    task_factory=create_agent,
)
```

## When Do I Need a Factory?

| Scenario | `task=` | `task_factory=` |
|----------|:-------:|:---------------:|
| Pure function, no side effects | Yes | |
| Stateless API wrapper | Yes | |
| Agent with message history | | Yes |
| Agent with mutable config/state | | Yes |
| Database connection per run | | Yes |
| `repeat=1` (any task) | Yes | |

**Rule of thumb:** If your task modifies `self` or any mutable state between calls, use `task_factory`.

## Resource Cleanup

If your factory creates resources that need cleanup (database connections, file handles), handle it inside the task:

```python
def create_agent():
    db = connect_to_database()

    async def task(question: str) -> str:
        try:
            result = await query_agent(db, question)
            return result
        finally:
            db.close()

    return task
```

## Validation

`task` and `task_factory` are mutually exclusive. Providing both (or neither) raises a `ValueError`:

```python
# Raises ValueError: "Provide either 'task' or 'task_factory', not both."
await evaluate_testset_with_mlflow(testset, task=fn, task_factory=factory)

# Raises ValueError: "Provide either 'task' or 'task_factory'."
await evaluate_testset_with_mlflow(testset)
```
