# Attempt 53 protocol — Bayesian and hybrid optimization in the rank-15 space

Date frozen: 2026-07-30

Evidence status before execution: exploratory protocol only

Implementation amendment before any calibration result: the first smoke
invocation stopped before constructing a truth because the shared
`phase3_common.make_truth` intentionally rejects seeds outside Attempts
35–49. Attempt 53 therefore reproduces the same frozen
`SeedSequence([113, 11, seed])`, control-map, and Hermitian-drift recipe in a
local wrapper that accepts only the four seeds listed below. The shared truth
generator and all earlier result files remain unchanged. No method query or
fidelity result existed when this amendment was made.

## Question

Does noisy Bayesian optimization, or Bayesian warm-start followed by one
Hessian-preconditioned local update, improve finite-shot calibration relative
to the existing principal-global method or a budget-matched sequential line
scan?

This is a synthetic development experiment. It does not modify the completed
Attempt-49 confirmation and cannot strengthen the submitted claim without a
separate preregistered holdout.

## Fixed physical and black-box setup

- synthetic two-qubit CNOT benchmark;
- 40 real pulse parameters;
- the same differentiable nominal model and top 15 positive-curvature Hessian
  directions used by Attempts 44 and 49;
- query-only mismatched finite-shot device;
- mismatch families `control-map`, `drift`, and `combined`;
- fixed mismatch magnitudes 0.05, 0.10, and 0.05 respectively;
- truth seeds 260701–260704 and two noise replicates per truth cell; and
- target infidelity `1e-3`.

The optimizer receives only sampled scalar fidelity and the known nominal
rank-15 basis/curvatures. Exact truth fidelity and truth derivatives are
available only after the client closes.

## Equal resource budget

Methods use one of three frozen resource budgets:

| Budget | Decision queries at 32768 shots | Sentinels | Total queries | Total shots |
|---|---:|---:|---:|---:|
| reduced | 32 | 2 × 1024 shots | 34 | 1,050,624 |
| hybrid-reduced | 48 | 2 × 1024 shots | 50 | 1,574,912 |
| full | 64 | 2 × 1024 shots | 66 | 2,099,200 |

Equal-budget comparisons use identical totals. Reduced-budget methods are
compared against the full two-cycle local baseline to test whether measurement
can actually be reduced. The same seeded binomial stream is paired across
methods within each truth-cell/replicate unit. All runs are retained. There is
no result-dependent retry, seed replacement, or budget extension.

## Frozen methods

### `principal-global-1cycle-k15` and `principal-global-2cycle-k15`

The existing Attempt-49 method with either one reduced-budget cycle or two
full-budget cycles. Each cycle uses 30 central-difference queries plus
current/proposal validation. The step is preconditioned by the nominal Hessian
curvature plus the existing common ridge and clipped to radius 0.25.

### `sequential-quadratic-k15`

A budget-matched proxy for one-direction-at-a-time calibration, not a literal
reproduction of the paper's undisclosed scan-point allocation. It performs
one ordered pass over the 15 principal directions. Each line receives four
points at normalized offsets `[-1, -1/3, 1/3, 1]`, uses the existing
curvature-derived scan width, and fits a quadratic with best-measured fallback.
Two start and two final measurements consume the remaining four decision
queries; the final pulse is accepted only when their aggregate measured
improvement is positive at one-sided confidence 0.995.

### `bayes-32-k15` and `bayes-64-k15`

Noisy Bayesian optimization acts only on the 15 principal coefficients in a
radius-0.5 ball around the nominal pulse, matching the maximum displacement
reachable by two radius-0.25 local cycles.

- either 30 or 62 BO observations: zero plus 11 deterministic Sobol-ball
  points, followed by expected-improvement selections;
- a fixed Matérn-5/2 Gaussian process with length scales set once from nominal
  Hessian curvature and clipped to [0.05, 0.30];
- binomial observation variances from the measured fidelity and shot count;
- deterministic global and incumbent-centered candidate pools; and
- recommendation by minimum posterior mean among already measured points.

The final two decision queries validate start versus recommendation. Failed
validation returns the start pulse. Both methods share the same first 30 BO
observations.

### `bayes16-then-global-k15` and `bayes32-then-global-k15`

- either 14 or 30 BO observations, sharing the same frozen initial and
  acquisition history with the pure BO methods;
- two queries validate the BO recommendation against the nominal start;
- 30 central-difference queries estimate all 15 local derivatives from the
  accepted BO incumbent; and
- two queries validate one Hessian-preconditioned radius-0.25 local proposal.

The 14-observation hybrid uses the 50-query hybrid-reduced budget. The
30-observation hybrid uses the 66-query full budget.

## Outcome and interpretation

Primary exploratory outcome is whether the single online-selected terminal
output has post-hoc exact infidelity at most `1e-3`. Secondary outcomes include
best-accepted exact infidelity and family-stratified paired success
differences. BO receives no credit for a queried point that it did not select
as its terminal output.

The hidden exact value scores outcomes only after calibration. It never
selects a BO point, tunes a kernel, accepts a proposal, stops a run, excludes a
cell, or changes a hyperparameter.

An equal-budget method is promising only if it improves paired success. A
reduced-budget method can support a future "fewer measurements" claim only if
it remains non-inferior to the full two-cycle local baseline while using
strictly fewer queries and shots. Any promising choice must be frozen and
rerun on disjoint truth seeds before it is considered submission evidence.
