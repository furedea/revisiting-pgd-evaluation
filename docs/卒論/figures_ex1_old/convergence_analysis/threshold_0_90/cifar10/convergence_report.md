# Convergence Analysis Summary (CIFAR10)

**Dataset**: CIFAR10

**Threshold**: 90% of max loss

**Data points per init** (single sample):
- clean: 1 (deterministic, single point)
- random: 20 (stochastic)
- deepfool: 1 (deterministic, single point)
- multi_deepfool: 9 (deterministic, 9 target classes)

**NC Types**:
- Never Reached (NR): Never reached threshold
- Unstable (US): Reached threshold but not stable in final window

## Detailed Results (by model and init)

| Model | Init | Conv | Mean | Median | P95 | Max | N | NR | US |
|-------|------|-----:|-----:|-------:|----:|----:|--:|---:|---:|
| nat | clean | 100%(1/1) | 12.0 | 12.0 | 12.0 | 12 | 1 | 0%(0) | 0%(0) |
| nat | random | 100%(20/20) | 14.3 | 13.0 | 22.0 | 22 | 20 | 0%(0) | 0%(0) |
| nat | deepfool | 100%(1/1) | 12.0 | 12.0 | 12.0 | 12 | 1 | 0%(0) | 0%(0) |
| nat | multi_deepfool | 100%(9/9) | 12.9 | 12.0 | 20.0 | 24 | 9 | 0%(0) | 0%(0) |
| nat_and_adv | clean | 100%(1/1) | 5.0 | 5.0 | 5.0 | 5 | 1 | 0%(0) | 0%(0) |
| nat_and_adv | random | 100%(20/20) | 7.0 | 7.0 | 7.0 | 7 | 20 | 0%(0) | 0%(0) |
| nat_and_adv | deepfool | 100%(1/1) | 4.0 | 4.0 | 4.0 | 4 | 1 | 0%(0) | 0%(0) |
| nat_and_adv | multi_deepfool | 100%(9/9) | 6.3 | 6.0 | 7.0 | 7 | 9 | 0%(0) | 0%(0) |
| weak_adv | clean | 100%(1/1) | 6.0 | 6.0 | 6.0 | 6 | 1 | 0%(0) | 0%(0) |
| weak_adv | random | 100%(20/20) | 7.7 | 8.0 | 8.0 | 8 | 20 | 0%(0) | 0%(0) |
| weak_adv | deepfool | 100%(1/1) | 5.0 | 5.0 | 5.0 | 5 | 1 | 0%(0) | 0%(0) |
| weak_adv | multi_deepfool | 100%(9/9) | 6.0 | 6.0 | 7.0 | 7 | 9 | 0%(0) | 0%(0) |
| adv | clean | 100%(1/1) | 8.0 | 8.0 | 8.0 | 8 | 1 | 0%(0) | 0%(0) |
| adv | random | 100%(20/20) | 9.7 | 10.0 | 10.0 | 10 | 20 | 0%(0) | 0%(0) |
| adv | deepfool | 100%(1/1) | 6.0 | 6.0 | 6.0 | 6 | 1 | 0%(0) | 0%(0) |
| adv | multi_deepfool | 100%(9/9) | 10.4 | 11.0 | 12.6 | 13 | 9 | 0%(0) | 0%(0) |

## Model-level Summary

| Model | Conv | Mean | Median | P95 | Max | N | NR | US |
|-------|-----:|-----:|-------:|----:|----:|--:|---:|---:|
| nat | 100%(31/31) | 13.7 | 12.0 | 22.0 | 24 | 31 | 0%(0) | 0%(0) |
| nat_and_adv | 100%(31/31) | 6.6 | 7.0 | 7.0 | 7 | 31 | 0%(0) | 0%(0) |
| weak_adv | 100%(31/31) | 7.1 | 7.0 | 8.0 | 8 | 31 | 0%(0) | 0%(0) |
| adv | 100%(31/31) | 9.7 | 10.0 | 11.5 | 13 | 31 | 0%(0) | 0%(0) |

## Overall Summary (All Models Combined)

| Metric | Value |
|--------|------:|
| Convergence Rate | 100%(124/124) |
| Total samples | 124 |
| Converged | 124 |
| NC (Never Reached) | 0%(0) |
| NC (Unstable) | 0%(0) |
| Mean (converged) | 9.3 |
| Median (converged) | 8.0 |
| P95 (converged) | 15.8 |
| Max (converged) | 24 |

## Recommendation

- **95% of converged samples** reach 90% of max loss by iteration **16**
- **99% of converged samples** reach 90% of max loss by iteration **22**
