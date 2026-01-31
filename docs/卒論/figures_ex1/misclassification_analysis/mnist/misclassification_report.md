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
| nat_and_adv | random | 0%(0/20) | N/A | N/A | N/A | N/A | 20 | 100%(20) |
| nat_and_adv | deepfool | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 100%(1) |
| nat_and_adv | multi_deepfool | 0%(0/9) | N/A | N/A | N/A | N/A | 9 | 100%(9) |
| adv | random | 0%(0/20) | N/A | N/A | N/A | N/A | 20 | 100%(20) |
| adv | deepfool | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 100%(1) |
| adv | multi_deepfool | 0%(0/9) | N/A | N/A | N/A | N/A | 9 | 100%(9) |
| weak_adv | random | 100%(20/20) | 25.5 | 25.0 | 29.1 | 30 | 20 | 0%(0) |
| weak_adv | deepfool | 100%(1/1) | 22.0 | 22.0 | 22.0 | 22 | 1 | 0%(0) |
| weak_adv | multi_deepfool | 100%(9/9) | 28.1 | 28.0 | 33.0 | 35 | 9 | 0%(0) |

## Model-level Summary

| Model | Attack Rate | Mean | Median | P95 | Max | N | Failed |
|-------|------------:|-----:|-------:|----:|----:|--:|-------:|
| nat | 100%(31/31) | 4.8 | 6.0 | 8.0 | 12 | 31 | 0%(0) |
| nat_and_adv | 0%(0/31) | N/A | N/A | N/A | N/A | 31 | 100%(31) |
| adv | 0%(0/31) | N/A | N/A | N/A | N/A | 31 | 100%(31) |
| weak_adv | 100%(31/31) | 26.5 | 26.0 | 32.5 | 35 | 31 | 0%(0) |

## Overall Summary (All Models Combined)

| Metric | Value |
|--------|------:|
| Attack Success Rate | 50%(60/120) |
| Total samples | 120 |
| Misclassified | 60 |
| Failed (never misclassified) | 50%(60) |
| Mean iteration | 15.4 |
| Median iteration | 15.0 |
| P95 iteration | 30.0 |
| Max iteration | 35 |

## Key Findings

- **50% of successful attacks** achieve misclassification by iteration **15**
- **90% of successful attacks** achieve misclassification by iteration **29**
- **95% of successful attacks** achieve misclassification by iteration **30**
