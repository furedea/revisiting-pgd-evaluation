# Misclassification Speed Analysis Summary (CIFAR10)

**Dataset**: CIFAR10

**Metric**: First misclassification iteration (attack success speed)

**Data points per init** (single sample):
- clean: 1 (deterministic, single point)
- random: 20 (stochastic)
- deepfool: 1 (deterministic, single point)
- multi_deepfool: 9 (deterministic, 9 target classes)

## Detailed Results (by model and init)

| Model | Init | Attack Rate | Mean | Median | P95 | Max | N | Failed |
|-------|------|------------:|-----:|-------:|----:|----:|--:|-------:|
| nat | clean | 100%(1/1) | 1.0 | 1.0 | 1.0 | 1 | 1 | 0%(0) |
| nat | random | 100%(20/20) | 1.3 | 1.0 | 2.0 | 2 | 20 | 0%(0) |
| nat | deepfool | 100%(1/1) | 0.0 | 0.0 | 0.0 | 0 | 1 | 0%(0) |
| nat | multi_deepfool | 100%(9/9) | 0.3 | 0.0 | 1.0 | 1 | 9 | 0%(0) |
| nat_and_adv | clean | 100%(1/1) | 1.0 | 1.0 | 1.0 | 1 | 1 | 0%(0) |
| nat_and_adv | random | 100%(20/20) | 1.0 | 1.0 | 1.0 | 1 | 20 | 0%(0) |
| nat_and_adv | deepfool | 100%(1/1) | 0.0 | 0.0 | 0.0 | 0 | 1 | 0%(0) |
| nat_and_adv | multi_deepfool | 100%(9/9) | 0.0 | 0.0 | 0.0 | 0 | 9 | 0%(0) |
| adv | clean | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 100%(1) |
| adv | random | 0%(0/20) | N/A | N/A | N/A | N/A | 20 | 100%(20) |
| adv | deepfool | 0%(0/1) | N/A | N/A | N/A | N/A | 1 | 100%(1) |
| adv | multi_deepfool | 0%(0/9) | N/A | N/A | N/A | N/A | 9 | 100%(9) |
| weak_adv | clean | 100%(1/1) | 2.0 | 2.0 | 2.0 | 2 | 1 | 0%(0) |
| weak_adv | random | 100%(20/20) | 2.0 | 2.0 | 2.0 | 2 | 20 | 0%(0) |
| weak_adv | deepfool | 100%(1/1) | 1.0 | 1.0 | 1.0 | 1 | 1 | 0%(0) |
| weak_adv | multi_deepfool | 100%(9/9) | 0.1 | 0.0 | 0.6 | 1 | 9 | 0%(0) |

## Model-level Summary

| Model | Attack Rate | Mean | Median | P95 | Max | N | Failed |
|-------|------------:|-----:|-------:|----:|----:|--:|-------:|
| nat | 100%(31/31) | 1.0 | 1.0 | 2.0 | 2 | 31 | 0%(0) |
| nat_and_adv | 100%(31/31) | 0.7 | 1.0 | 1.0 | 1 | 31 | 0%(0) |
| adv | 0%(0/31) | N/A | N/A | N/A | N/A | 31 | 100%(31) |
| weak_adv | 100%(31/31) | 1.4 | 2.0 | 2.0 | 2 | 31 | 0%(0) |

## Overall Summary (All Models Combined)

| Metric | Value |
|--------|------:|
| Attack Success Rate | 75%(93/124) |
| Total samples | 124 |
| Misclassified | 93 |
| Failed (never misclassified) | 25%(31) |
| Mean iteration | 1.0 |
| Median iteration | 1.0 |
| P95 iteration | 2.0 |
| Max iteration | 2 |

## Key Findings

- **50% of successful attacks** achieve misclassification by iteration **1**
- **90% of successful attacks** achieve misclassification by iteration **2**
- **95% of successful attacks** achieve misclassification by iteration **2**
