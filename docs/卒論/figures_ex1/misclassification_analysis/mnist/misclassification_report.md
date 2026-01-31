# Misclassification Speed Analysis Summary (MNIST)

**Dataset**: MNIST

**Metric**: First misclassification iteration (attack success speed)

**Data points per init** (single sample):
- clean: 1 (deterministic, single point)
- random: 20 (stochastic)
- deepfool: 1 (deterministic, single point)
- multi_deepfool: 9 (deterministic, 9 target classes)

## Detailed Results (by model and init)

| Model | Init | Attack Rate | Mean | Median | P95 | Max | N | Failed |
|-------|------|------------:|-----:|-------:|----:|----:|--:|-------:|
| nat | random | 100%(20/20) | 6.6 | 7.0 | 8.0 | 8 | 20 | 0%(0) |
| nat | deepfool | 100%(1/1) | 0.0 | 0.0 | 0.0 | 0 | 1 | 0%(0) |
| nat | multi_deepfool | 100%(9/9) | 0.7 | 1.0 | 1.0 | 1 | 9 | 0%(0) |
| nat_and_adv | random | 100%(20/20) | 0.0 | 0.0 | 0.0 | 0 | 20 | 0%(0) |
| nat_and_adv | deepfool | 100%(1/1) | 0.0 | 0.0 | 0.0 | 0 | 1 | 0%(0) |
| nat_and_adv | multi_deepfool | 100%(9/9) | 0.0 | 0.0 | 0.0 | 0 | 9 | 0%(0) |
| adv | random | 100%(20/20) | 0.0 | 0.0 | 0.0 | 0 | 20 | 0%(0) |
| adv | deepfool | 100%(1/1) | 0.0 | 0.0 | 0.0 | 0 | 1 | 0%(0) |
| adv | multi_deepfool | 100%(9/9) | 0.0 | 0.0 | 0.0 | 0 | 9 | 0%(0) |
| weak_adv | random | 100%(20/20) | 25.5 | 25.0 | 29.1 | 30 | 20 | 0%(0) |
| weak_adv | deepfool | 100%(1/1) | 22.0 | 22.0 | 22.0 | 22 | 1 | 0%(0) |
| weak_adv | multi_deepfool | 100%(9/9) | 28.1 | 28.0 | 33.0 | 35 | 9 | 0%(0) |

## Model-level Summary

| Model | Attack Rate | Mean | Median | P95 | Max | N | Failed |
|-------|------------:|-----:|-------:|----:|----:|--:|-------:|
| nat | 100%(31/31) | 4.8 | 6.0 | 8.0 | 12 | 31 | 0%(0) |
| nat_and_adv | 100%(31/31) | 0.0 | 0.0 | 0.0 | 0 | 31 | 0%(0) |
| adv | 100%(31/31) | 0.0 | 0.0 | 0.0 | 0 | 31 | 0%(0) |
| weak_adv | 100%(31/31) | 26.5 | 26.0 | 32.5 | 35 | 31 | 0%(0) |

## Overall Summary (All Models Combined)

| Metric | Value |
|--------|------:|
| Attack Success Rate | 100%(120/120) |
| Total samples | 120 |
| Misclassified | 120 |
| Failed (never misclassified) | 0%(0) |
| Mean iteration | 7.7 |
| Median iteration | 0.0 |
| P95 iteration | 29.0 |
| Max iteration | 35 |

## Key Findings

- **50% of successful attacks** achieve misclassification by iteration **0**
- **90% of successful attacks** achieve misclassification by iteration **27**
- **95% of successful attacks** achieve misclassification by iteration **29**
