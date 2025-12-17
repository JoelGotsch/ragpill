# ADR-0018: Gateable metrics are conservative; infra failures fail closed

**Status:** Accepted
**Date:** 2026-07-07
**Impact:** Medium
**Related:** ADR-0011, ADR-0012 (superseded by this ADR)

## Context

Phase 2 introduced honest failure attribution: a run whose trace never arrived
is an *error state* (`RunResult.is_error_state`), distinct from a fail verdict.
That raised the question ADR-0012 could not have answered in June (the
infra/verdict split did not exist yet): what do error-state runs do to pass
rates and verdicts?

Excluding them from denominators everywhere is honest as a metric but
**fail-open as a gate**: with 8 of 10 runs infra-errored and the 2 survivors
passing, `pass_rate = 1.0` and a threshold gate promotes the agent on a sample
that silently shrank by 80%. Exclusion can even correlate with agent quality —
an agent that hangs in ways that also break trace export gets its worst runs
excluded preferentially. Counting error runs as failures everywhere
(ADR-0012's arithmetic) is fail-closed but conflates infra outages with
quality regressions, which mis-directs debugging and breeds alarm fatigue.

The verdict-based gate alone cannot carry the safety property, because teams
gate promotion pipelines on whatever scalar lands in the tracking server, not
on the verdict object.

## Decision

Split gateable numbers from diagnostic numbers. **Any number a pipeline can
plausibly gate on is conservative by default: an infra failure may never
improve a reported result.**

- `pass_rate` (headline: aggregates, uploads, `summary` DataFrame):
  **conservative lower bound** — passing runs over *all* runs, error-state
  runs counted as non-passes. Infra trouble can only push it down; gating on
  it is fail-closed even for consumers that never read the verdict.
- `pass_rate_evaluated` (diagnostic): passing runs over evaluated runs.
  Never travels alone — always paired with `runs_evaluated` /
  `runs_infra_error`. The `_evaluated` suffix marks it as not-for-gating.
- The verdict: `passed = pass_rate >= threshold AND runs_infra_error == 0`.
  Any nonzero infra tolerance must be an explicit opt-in knob (none exists
  today), never a default. The two failure causes produce distinct summary
  text ("insufficient evaluated coverage" vs. accuracy below threshold).
- Per-evaluator: `per_evaluator_pass_rates` stays evaluated-only (diagnostic,
  read with `error_counts`); any per-evaluator number that is uploaded as a
  gateable verdict (the `agg_*` assessments, the cases-DataFrame `passed`
  column) is conservative and blocked by any errored run.
- `RunResult.all_passed` is `False` for error-state runs (fail-closed at run
  granularity), expressed via `is_error_state` so the tri-state
  (verdict / task error / infra error) has one encoding.

## Alternatives considered

- **Evaluated-only denominators everywhere ("pure exclusion").** Rejected:
  fail-open for metric-gating consumers, per the scenario above.
- **Error runs count as failures everywhere (ADR-0012's arithmetic).**
  Rejected: destroys attribution — a backend blip is indistinguishable from a
  quality regression, unwinding Phase 2's purpose.
- **Keep ADR-0012's freeze on `all_passed`.** Rejected: its stated concern
  (threshold stability — "shifts reported pass rates") is honored in the only
  direction that matters, since the conservative definition can only move
  rates *down* on infra failure, making gates stricter, never looser.

## Consequences

- ADR-0012 is superseded. Its triage-visibility decision (ERROR lines,
  failures never invisible) is retained; its `all_passed` freeze and 0-pass
  arithmetic are replaced by the conservative/evaluated split.
- Reported pass rates change for runs with evaluator errors (they drop).
  Announced in the CHANGELOG under Breaking.
- Every reporting surface (aggregates, uploads, DataFrames, triage) shows the
  conservative number as `pass_rate` and carries coverage counts next to any
  evaluated-only number.
- A case with any infra-degraded run cannot pass, regardless of accuracy on
  the evaluated remainder.
