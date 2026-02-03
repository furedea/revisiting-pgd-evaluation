# Convergence Analysis Summary (MNIST)

**Dataset**: MNIST

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
| nat | clean | 100%(1/1) | 45.0 | 45.0 | 45.0 | 45 | 1 | 0%(0) | 0%(0) |
| nat | random | 100%(20/20) | 48.5 | 47.5 | 55.1 | 58 | 20 | 0%(0) | 0%(0) |
| nat | deepfool | 100%(1/1) | 33.0 | 33.0 | 33.0 | 33 | 1 | 0%(0) | 0%(0) |
| nat | multi_deepfool | 100%(9/9) | 39.6 | 40.0 | 43.6 | 44 | 9 | 0%(0) | 0%(0) |
| nat_and_adv | clean | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 0%(0) | 100%(1) |
| nat_and_adv | random | 95%(19/20) | 60.7 | 60.0 | 76.5 | 81 | 20 | 0%(0) | 5%(1) |
| nat_and_adv | deepfool | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 0%(0) | 100%(1) |
| nat_and_adv | multi_deepfool | 11%(1/9) | 52.0 | 52.0 | 52.0 | 52 | 9 | 0%(0) | 89%(8) |
| weak_adv | clean | 100%(1/1) | 83.0 | 83.0 | 83.0 | 83 | 1 | 0%(0) | 0%(0) |
| weak_adv | random | 100%(20/20) | 73.2 | 71.0 | 85.0 | 85 | 20 | 0%(0) | 0%(0) |
| weak_adv | deepfool | 100%(1/1) | 87.0 | 87.0 | 87.0 | 87 | 1 | 0%(0) | 0%(0) |
| weak_adv | multi_deepfool | 78%(7/9) | 85.0 | 86.0 | 87.7 | 88 | 9 | 0%(0) | 22%(2) |
| adv | clean | 100%(1/1) | 41.0 | 41.0 | 41.0 | 41 | 1 | 0%(0) | 0%(0) |
| adv | random | 100%(20/20) | 54.0 | 50.0 | 84.2 | 88 | 20 | 0%(0) | 0%(0) |
| adv | deepfool | 100%(1/1) | 49.0 | 49.0 | 49.0 | 49 | 1 | 0%(0) | 0%(0) |
| adv | multi_deepfool | 100%(9/9) | 55.9 | 52.0 | 71.2 | 78 | 9 | 0%(0) | 0%(0) |

## Model-level Summary

| Model | Conv | Mean | Median | P95 | Max | N | NR | US |
|-------|-----:|-----:|-------:|----:|----:|--:|---:|---:|
| nat | 100%(31/31) | 45.3 | 45.0 | 55.0 | 58 | 31 | 0%(0) | 0%(0) |
| nat_and_adv | 65%(20/31) | 60.2 | 59.5 | 76.2 | 81 | 31 | 0%(0) | 35%(11) |
| weak_adv | 94%(29/31) | 76.8 | 77.0 | 87.0 | 88 | 31 | 0%(0) | 6%(2) |
| adv | 100%(31/31) | 54.0 | 51.0 | 81.0 | 88 | 31 | 0%(0) | 0%(0) |

## Overall Summary (All Models Combined)

| Metric | Value |
|--------|------:|
| Convergence Rate | 90%(111/124) |
| Total samples | 124 |
| Converged | 111 |
| NC (Never Reached) | 0%(0) |
| NC (Unstable) | 10%(13) |
| Mean (converged) | 58.7 |
| Median (converged) | 56.0 |
| P95 (converged) | 85.5 |
| Max (converged) | 88 |

## Recommendation

- **95% of converged samples** reach 90% of max loss by iteration **86**
- **99% of converged samples** reach 90% of max loss by iteration **88**
