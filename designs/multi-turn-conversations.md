# Design: Multi-Turn Conversation Evaluation

**Status:** Draft
**Date:** 2026-06-19
**Related:**
[layered-execution-evaluation.md](layered-execution-evaluation.md) (the
execute/evaluate/upload split this builds on),
[langfuse-integration.md](langfuse-integration.md) §9.1 (a Langfuse-framed
sketch of scripted multi-turn via generic `Conversation`/`ConversationOutput`
types — superseded/extended here),
[../plans/repeat-task-runs.md](../plans/repeat-task-runs.md) (the repeat
model, deliberately *not* the turn model),
[../plans/sessions-for-case-grouping.md](../plans/sessions-for-case-grouping.md)
(the session/case-grouping work this reuses for trace shape).

---

## 1. Goal

Let ragpill evaluate **multi-turn conversations**, not just single
`input → output` task runs. A conversation is a sequence of user messages and
agent replies that share state (the agent sees the history when answering turn
*k*). Two flavours must be supported:

1. **Scripted conversations** — the user turns are fixed strings authored in
   the dataset (e.g. a CSV row with `["What is X?", "And how does it compare to Y?"]`).
2. **Dynamic / user-agent conversations** — only an initial user goal/persona
   is authored; a configurable **user agent** (itself an LLM, configured the
   way `LLMJudge` is) reads the real agent's reply at turn *k* and *generates*
   the next user message for turn *k+1*. The conversation therefore branches
   on what the agent actually said, which is the realistic stress test for a
   RAG/chat system (clarifications, push-back, topic drift).

The design must slot into the existing three-layer pipeline (execute →
evaluate → upload) without forking it, and must leave today's single-turn
datasets working unchanged.

## 2. Non-goals

- **Replacing `repeat`.** `repeat` (N independent re-runs of one case for
  stochasticity) is orthogonal to turns and stays exactly as-is. A multi-turn
  case can still be repeated N times (§3).
- **Tree/branching exploration** beyond a single linear path per run. Each run
  produces one conversation thread; we do not fan out one agent reply into
  multiple alternative user follow-ups in a single run (that's a future
  "conversation tree" extension).
- **Live human-in-the-loop conversations.** The user side is either scripted
  or LLM-simulated; no interactive prompt.
- **Cross-case memory.** Conversations are isolated from each other, like cases
  are today.
- **A new backend/UI.** We reuse the session/case-grouping mechanism already in
  `execution.py` and the backends.

## 3. Terminology

These three axes are independent and must not be conflated.

| Term | Meaning | Cardinality | Existing? |
|------|---------|-------------|-----------|
| **Case** | One dataset entry: an input (or conversation spec), expected output, metadata, evaluators. | 1 per dataset row | yes (`Case`) |
| **Repeat** | An *independent re-run* of the same case to measure stochasticity. Repeats share nothing; a fresh task instance per repeat. Pass/fail by `threshold`. | N per case (`repeat`) | yes (`TestCaseMetadata.repeat`) |
| **Turn** | One *message exchange within a single conversation*: a user message → an agent reply. Turns are **stateful** — turn *k* sees turns `0..k-1`. | M per repeat | **new** |

The mental model is a 3-level nesting:

```
Case "compare X and Y"
└── repeat 0                         (independent re-run)
│   ├── turn 0  user:"What is X?"      agent:"X is …"
│   ├── turn 1  user:"How vs Y?"       agent:"Compared to Y …"
│   └── turn 2  user:"Thanks, source?" agent:"See doc#3 …"
└── repeat 1                         (independent re-run, may branch differently
    ├── turn 0  …                      under a user agent because the agent's
    └── …                              turn-0 reply differs)
```

Crucially: **turns are not repeats**. A repeat reruns the whole conversation
from scratch; a turn advances one conversation forward by one message. Under a
user agent the *number* and *content* of turns in repeat 0 and repeat 1 may
differ, because the user agent reacts to a stochastic agent.

## 4. Context: how single-turn works today

- `execute_dataset` (in `execution.py`) loops cases → for each case calls
  `_execute_case_runs`, which opens a **case grouping** (`start_case_grouping`)
  and loops `repeat` times calling `_execute_single_run`.
- `_execute_single_run` opens one span/trace per repeat, calls
  `await task(case.inputs)` **once**, records output/duration/error, and
  returns a `TaskRunOutput`.
- Trace capture has two modes already (this is the key reuse point):
  - **session mode** (MLflow with `mlflow.trace.session`, Langfuse with
    `session_id`): each repeat is its *own top-level trace*, all tagged into one
    session named after the case. `CaseRunOutput.trace` is `None`; each
    `TaskRunOutput.trace` is that repeat's trace.
  - **span mode** (fallback): the case opens a parent span, repeats nest under
    it, and the post-loop fetches the case trace once and filters per-repeat
    subtrees via `filter_to_subtree`.
- Evaluators (`evaluators.py`) receive an `EvaluatorContext` with `inputs`,
  `output`, `trace`, and `run_span_id`. `LLMJudge` reads `ctx.inputs` /
  `ctx.output`; `SpanBaseEvaluator` subclasses walk `ctx.trace` (scoped to
  `ctx.run_span_id`) to inspect retriever/tool spans.
- The user-agent's LLM should be configured the way `LLMJudge`'s is: a
  `LLMJudgeSettings`-style settings object that lazily builds and caches a
  `pydantic_ai.models.Model` (`base_url` / `api_key` / `model_name` /
  SSL knobs), injectable via a `set_model` / `configure_*` singleton.

The signature `TaskType = Callable[[Any], Awaitable[Any]]` is the constraint:
it takes a *single* input. Multi-turn needs the task to be called repeatedly
with growing history. That is the central integration problem.

## 5. Data model

### 5.1 Conversation spec (the case input)

A multi-turn case is still a `Case[InputsT, OutputT, MetadataT]`; we only
specialise `InputsT`. Introduce a `ragpill.multi_turn` module:

```python
@dataclass
class UserTurn:
    """One authored (scripted) user message."""
    content: str

@dataclass
class UserAgentSpec:
    """Config for a simulated user that generates follow-ups dynamically."""
    goal: str                         # what the simulated user is trying to achieve
    persona: str | None = None        # optional style/role ("impatient novice")
    max_turns: int = 4                # hard cap on conversation length
    stop_when: str | None = None      # natural-language stop condition for the user agent
    model: Any | None = None          # optional per-case model override (else global)

@dataclass
class Conversation:
    """A multi-turn case input.

    Exactly one of `scripted` or `user_agent` drives the user side.
    `seed` is the opening user message (required for user_agent; for scripted
    it is just turns[0]).
    """
    scripted: list[UserTurn] | None = None
    user_agent: UserAgentSpec | None = None
    seed: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

A plain `str` input (today's case) is treated as a single-turn conversation —
no migration needed.

### 5.2 Conversation output

```python
@dataclass
class Turn:
    index: int
    user_message: str          # scripted text, or what the user agent produced
    agent_output: Any          # the task's reply for this turn
    duration: float
    error: str | None = None

@dataclass
class ConversationOutput:
    turns: list[Turn]
    final_output: Any          # convenience: turns[-1].agent_output
    stopped_reason: str        # "max_turns" | "stop_condition" | "scripted_exhausted" | "error"
```

`ConversationOutput` is what `TaskRunOutput.output` holds for a multi-turn
case, so JSON round-trip (`_task_run_to_dict`) needs to serialise it (it is
plain dataclasses → trivially `asdict`-able; add a typed (de)serialiser
alongside the trace helpers).

### 5.3 How conversation history is carried to the task

The task can no longer be `fn(input) -> output`. We need it to see history.
Two task shapes are supported, mirroring the existing `task` vs `task_factory`
split:

- **Stateless turn function:** `fn(user_message, history) -> reply`, where
  `history: list[Turn]`. ragpill owns the loop and threads history in. Best
  for HTTP-style agents that accept a message list.
- **Stateful agent factory:** `factory() -> agent`, where `agent` is called
  once per turn and keeps its own memory between calls within one conversation.
  A fresh `agent` is built per **repeat** (so repeats stay independent — same
  isolation guarantee `task_factory` gives today). Within a repeat, the same
  `agent` instance is reused across turns.

Both are adapted into the existing `TaskType` by a wrapper the executor builds,
so `execute_dataset`'s public surface barely changes (§7).

### 5.4 Trace capture mapping

This is where the existing **session/case-grouping** work pays off directly.
The natural mapping is:

- **one session = one conversation** (one repeat of a multi-turn case)
- **one trace = one turn**

In **session-mode** backends we already open one trace per repeat tagged into a
case session. For multi-turn we go one level deeper: open one trace **per turn**
tagged into a *conversation* session. The session id becomes
`f"ragpill-{base_key}_{run_index}"` (case + repeat), and each turn's trace
carries `metadata={"turn_index": k}`. This is the threaded-conversation UI
Langfuse/MLflow sessions are built for, with zero new backend code beyond
passing a per-turn name/metadata.

In **span-mode** backends, a turn becomes a child span under the repeat's run
span (turns nest where repeats nest in span mode). `filter_to_subtree` already
lets evaluators scope to a span, so a per-turn span id gives per-turn trace
slices for free.

Concretely `TaskRunOutput` (one per repeat) grows an optional
`turn_traces: list[Trace | None]` (session mode) **or** the per-repeat `trace`
already holds all turn spans and we filter per turn by `turn_span_ids`
(span mode) — symmetric with how repeats are handled inside a case today.

## 6. The user agent

### 6.1 Configuration (model injection, like LLMJudge)

Add `UserAgentSettings(BaseSettings)` with `env_prefix="RAGPILL_USERAGENT_"`,
structurally identical to `LLMJudgeSettings`: `model_name`, `temperature`,
`base_url`, `api_key`, `ssl_ca_cert`, `ssl_verify`, a cached `llm_model`
property, a `set_model()` escape hatch, and a `configure_user_agent()` /
`get_user_agent_settings()` singleton pair. A per-case
`UserAgentSpec.model` overrides the global singleton, exactly like passing a
`model=` to `judge_output`. This reuses `_get_pydantic_ai_llm_model` so
corporate SSL/proxy setup is shared with the judge.

### 6.2 Prompting

The user agent is a `pydantic_ai.Agent` with a system prompt built from the
spec (persona + goal + stop instruction), and a user prompt that is the
**conversation so far** rendered as a transcript. It returns structured output:

```python
class UserAgentReply(BaseModel):
    message: str       # the next user turn (empty if done)
    done: bool         # the user agent decided the goal is satisfied
    reason: str        # why it asked this / why it's done
```

System prompt sketch (parity-with-`LLMJudge` style, prompt held as a module
constant):

> You are simulating a real user talking to an AI assistant. Your goal:
> {goal}. {persona}. Read the conversation so far and write the user's next
> message — a natural, context-dependent follow-up based on what the assistant
> just said. If your goal is fully satisfied {or: {stop_when}}, set `done:true`
> and leave `message` empty. Do not answer your own question; you are the user.

### 6.3 Termination

A conversation ends on the **first** of:

1. **`max_turns`** reached (hard cap; always set — protects against loops/cost).
2. **Stop condition** — `done:true` from the user agent (covers `stop_when`
   and "goal satisfied").
3. **Scripted exhaustion** — scripted mode ran out of authored turns.
4. **Error** — the task raised on a turn (record the error turn, stop).
5. *(optional, phase 2)* **Judge-decides** — an evaluator-style judge can mark
   the conversation complete (e.g. "the agent has answered everything"). Kept
   out of the core loop initially to avoid coupling execute → evaluate.

`stopped_reason` records which fired.

## 7. Evaluation

Evaluators run in the existing Phase-2 evaluate layer against
`EvaluatorContext`. Multi-turn introduces a per-turn vs whole-conversation
distinction.

- **Whole-conversation evaluators (default).** `ctx.inputs` is the
  `Conversation`, `ctx.output` is the `ConversationOutput`. Existing evaluators
  keep working with a small adaptation: `LLMJudge` should grade against the
  full transcript (render `ConversationOutput.turns` into the `<Output>`
  section). New conversation-level judges live in
  `ragpill.evaluators.conversation` (e.g. `ContextRetentionJudge`,
  `GoalAchievedJudge`, `TurnCountEvaluator`). One verdict per case per
  evaluator — fits the current report model with no schema change.
- **Per-turn evaluators.** Add a `per_turn: bool` (or a `TurnEvaluator` mixin)
  so an evaluator runs once per turn, producing one verdict per turn. The
  evaluate layer iterates `ConversationOutput.turns`, building a per-turn
  `EvaluatorContext` whose `output` is `turn.agent_output`, whose `trace` is
  that turn's trace (§5.4), and whose `run_span_id` scopes to the turn span.
  This is the **direct analogue** of how repeats are already aggregated, so
  the aggregation/threshold machinery can be reused (a per-turn pass rate
  vs a per-repeat pass rate).
- **Span-based evaluators see per-turn traces.** Because §5.4 gives each turn
  its own trace (session mode) or its own filtered subtree (span mode),
  `SpanBaseEvaluator`/`SourcesBaseEvaluator` work unchanged when run per-turn:
  set `ctx.run_span_id` to the turn span and they walk only that turn's
  retriever/tool spans. Run whole-conversation, they walk the union of all
  turns (today's behaviour against the case trace).

## 8. Implementation approaches

### Approach A — Generic types, executor owns the turn loop (recommended)

**Sketch.** No new `Case`/`EvaluatorContext` types. Specialise `InputsT =
Conversation`, `OutputT = ConversationOutput`. The executor detects a
`Conversation` input (or any input the user wraps with a `multi_turn(...)`
adapter) and, inside `_execute_single_run`, runs an inner **turn loop** instead
of a single `await task(input)`: for each turn it (a) gets the next user
message — from `scripted[k]` or by calling the user agent with the transcript —
(b) opens a per-turn trace/span, (c) calls the task with `(message, history)`
or the stateful agent, (d) appends a `Turn`. The loop terminates per §6.3 and
returns a `ConversationOutput` as the run's `output`.

**Reuse vs extend.**
- *Execute:* reuses `_execute_case_runs` (repeat loop, case grouping) verbatim;
  adds an inner turn loop and a `_run_conversation` helper. Reuses the existing
  session/span trace modes — turns map onto them the same way repeats do.
- *Evaluate:* reuses `EvaluatorContext` unchanged for whole-conversation; adds
  the optional per-turn iteration described in §7.
- *Upload:* reuses session upload; `split_conversations_into_sessions` from
  the Langfuse design (§9.1 there) becomes the natural default for multi-turn —
  one session per conversation, one trace per turn.

**Tradeoffs.** + Smallest blast radius; the three layers and all serialisation
keep their shapes. + Single-turn is literally the M=1 case. − The executor
grows real branching logic (scripted vs user-agent) — needs careful tests. −
`ConversationOutput` must be (de)serialised in the run-JSON helpers.

**Migration/compat.** Fully backward compatible: a `str`/non-`Conversation`
input takes the existing single-`await` path. Run-JSON `schema_version` stays 2
if we register a typed codec for `ConversationOutput`; otherwise bump to 3 with
a forward-only note (consistent with the ADR-0014 "re-run to upgrade" stance).

### Approach B — New first-class `ConversationCase` / `ConversationEvaluator`

**Sketch.** Add dedicated `ConversationCase`, `Turn`, and a
`ConversationEvaluatorContext` carrying `turns: list[Turn]` as a first-class
field, plus a parallel `execute_conversations()` entry point.

**Reuse vs extend.** Largely *parallel* code paths: a second executor, a second
context type, evaluator base classes that understand turns natively.

**Tradeoffs.** + Most explicit/discoverable API; turn semantics are typed, not
convention. + No overloading of `inputs`/`output` meaning. − Large surface-area
duplication of execute/evaluate/upload; two code paths to keep in sync. −
Pushes against the deliberate decision in the Langfuse design (§9.1) *not* to
add `SessionCase`/`Turn`/`SessionEvaluator`. − Heavier docs/test burden.

**Migration/compat.** Additive (old path untouched) but bifurcates the mental
model and the report code that already special-cases repeats.

### Approach C — Convention-only: user writes a stateful task, ragpill stays single-turn

**Sketch.** Do nothing to the core. Document a pattern where the user's *task*
internally runs the whole conversation (including instantiating their own user
agent) and returns a transcript string/object. ragpill sees one `input → output`
and evaluates the transcript with ordinary evaluators.

**Reuse vs extend.** Pure reuse; zero core changes. Ship only helper utilities
+ a guide.

**Tradeoffs.** + Zero risk, ships immediately. + Trace shape is already right
(N nested spans inside the run trace — exactly the Langfuse §9.1 argument). −
ragpill provides *no* user agent, no per-turn evaluation, no per-turn traces in
the report, no termination control — all reinvented per user. − The marquee
feature (a configured, branching user agent) lives outside the library, so it
can't be standardised, tested, or surfaced in the UI as turns.

**Migration/compat.** Nothing changes; but it doesn't really deliver the goal.

### Recommendation

**Approach A.** It delivers both scripted and user-agent conversations,
gives first-class per-turn traces and per-turn evaluation, and reuses the
session/case-grouping work and all three layers with the smallest, well-bounded
change (an inner turn loop + a `multi_turn` module + a `UserAgentSettings`
twin of `LLMJudgeSettings`). It is consistent with the existing Langfuse
design's explicit choice to model multi-turn via the generic `Case` parameters
rather than new case types (Approach B), while going beyond that sketch by
adding the dynamic user agent and per-turn granularity that the scripted-only
§9.1 lacks. Approach C can ship first as an interim/escape-hatch pattern but is
not the destination.

## 9. Open questions

1. **History format to the task.** Pass `list[Turn]`, an OpenAI-style
   `list[{role, content}]`, or pydantic-ai `ModelMessage`s? Probably offer a
   normaliser and let the adapter choose.
2. **Cost controls.** User-agent + agent + per-turn judges multiply token cost
   by turns × repeats. Need a global `max_turns` ceiling and possibly a
   per-run token budget that forces termination.
3. **Determinism / reproducibility.** A user agent at `temperature>0` makes
   repeats genuinely branch — desirable for stress, awkward for regression
   gates. Offer `temperature=0` + transcript pinning (record the realised
   scripted-equivalent so a branchy run can be replayed deterministically).
4. **Per-turn vs whole-conversation thresholds.** How does `threshold`
   compose across the repeat × turn grid? Likely: per-turn pass rate within a
   repeat, then per-repeat threshold across repeats — but the report columns
   need a decision.
5. **CSV authoring.** How are scripted turns / a user-agent goal expressed in a
   CSV row (the primary authoring path)? Probably a JSON-in-cell `check`-style
   column parsed by a `Conversation.from_csv_line`-style helper.
6. **User-agent loop guards.** Detect degenerate loops (user agent repeating
   itself, agent stalling) and stop early.

## 10. Suggested phased rollout

- **Phase 0 — interim (Approach C as a guide).** Publish the "task owns the
  conversation" pattern + a `Conversation`/`ConversationOutput` dataclass pair
  and `ContextRetentionJudge` that grade a transcript. Unblocks users now.
- **Phase 1 — scripted multi-turn in the core (Approach A, no user agent).**
  Executor turn loop for `scripted` conversations; per-turn traces via the
  session/span mapping (§5.4); `ConversationOutput` serialisation; whole-
  conversation evaluation. Single-turn path provably unchanged.
- **Phase 2 — the user agent.** `UserAgentSettings` (LLMJudge twin),
  `UserAgentSpec`, the simulated-user `pydantic_ai.Agent`, termination on
  `max_turns` / `done` / error. Dynamic branching conversations.
- **Phase 3 — per-turn evaluation + thresholds.** `per_turn` evaluators,
  per-turn span scoping for `SpanBaseEvaluator`, threshold composition across
  repeat × turn, report columns. CSV authoring helpers.
- **Phase 4 — polish.** Judge-decides termination, cost budgets, loop guards,
  deterministic replay/transcript pinning, Langfuse threaded-session upload as
  the multi-turn default.
