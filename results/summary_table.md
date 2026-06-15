# MNIST addition: Categories A, B, C summary

Shared CNN, seed 42, all runs local on Apple M2 (CPU). NeurASP uses a column-wise ASP
encoding; DeepProbLog uses its standard encoding with exact inference. Sum accuracy
drops as digits grow because every digit must be correct (0.975^6 is about 0.86),
while digit accuracy stays around 97%.

| Category / framework | Digits | Sum acc (%) | Digit acc (%) | Total time (s) | Iters to 90% sum (pairs) |
|---|---|---|---|---|---|
| NeurASP (Cat A) | 1 | 95.08 | 97.45 | 41.98 | 3000 |
| NeurASP (Cat A) | 2 | 92.52 | 98.07 | 94.23 | 9000 |
| NeurASP (Cat A) | 3 | 86.194 | 97.54 | 302.11 | n/a |
| DeepProbLog (Cat B) | 1 | 90.7 | 95.19 | 63.01 | 13500 |
| DeepProbLog (Cat B) | 2 | 91.2 | 97.64 | 468.27 | 15000 |
| LNN (Cat C) * | 1 | 92.7 | 97.11 | separate | n/a |

DeepProbLog-exact did not run at 3 digits: its inference circuit failed to compile
within 21 minutes. NeurASP completes 3 digits in about 5 minutes.

* LNN (Category C) is reasoning-only: it cannot backprop into a CNN, so the CNN was
trained separately on digit labels and LNN infers the sum from its predictions. This
is not directly comparable to the end-to-end A/B runs (CNN ~6.8 s, LNN ~63 ms/instance).
