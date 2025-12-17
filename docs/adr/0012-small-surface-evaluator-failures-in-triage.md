# ADR-0012: Surface evaluator failures in triage without changing `all_passed`

**Status:** Superseded by [ADR-0018](./0018-medium-conservative-gateable-metrics.md)
**Date:** 2026-06-19
**Impact:** Small
**Related:** ADR-0011, ADR-0018

> **Superseded 2026-07-07.** This ADR predates Phase 2's infra/verdict split
> (`trace_status`, `is_error_state`); once "the trace never arrived" became a
> first-class error state, freezing `all_passed` and counting error runs as
> 0-pass attempts stopped being coherent — see ADR-0018 for the replacement
> semantics (conservative gateable metrics, evaluated-only diagnostics with
> coverage). The triage-visibility decision below (ERROR lines; failures are
> never invisible) remains in force.

## Context
Evaluators that raised landed in `RunResult.evaluator_failures`, but the
triage report ignored that field entirely: `_per_evaluator_rollup` and
`_render_failing_run` only iterated `assertions`. As a result a run whose
assertions all passed but whose evaluators never actually ran (because they
errored) rendered as clean. This masked the trace-race failures (see
ADR-0011) — whole evaluators were being dropped with no signal in the report.

A proposed fix would additionally have made `RunResult.all_passed` return
`False` whenever `evaluator_failures` is non-empty.

## Decision
Surface `evaluator_failures` in the triage rollup — count each as a 0-pass
attempt, render an ERROR line per failed evaluator, and include runs whose
assertions passed but an evaluator errored. Do NOT change `all_passed`; it
intentionally continues to ignore `evaluator_failures`.

## Alternatives considered
- **Fold `evaluator_failures` into `all_passed` (return `False` on any
  evaluator error).** Rejected deliberately: it changes pass/fail semantics
  and shifts reported pass rates, which could move existing test thresholds.
  Making the failures visible in triage was judged sufficient without
  perturbing the verdict contract.

## Consequences
- A silently-dropped evaluator is now visible in the triage report (ERROR
  line, counted as a 0-pass attempt) without flipping the run's verdict.
- `all_passed` keeps its existing meaning; downstream thresholds and gating
  built on it are unaffected.
- Readers must remember that a run can be "passed" per `all_passed` while
  still showing evaluator ERROR lines in triage — the two answer different
  questions.

## References
- `RunResult.evaluator_failures`, `RunResult.all_passed`
- triage `_per_evaluator_rollup`, `_render_failing_run`
- Chat session 2026-06-19
