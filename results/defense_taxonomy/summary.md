# Defense Taxonomy and Generalization

Amplification factor A = 256; averaging budget R = 64.
Gains are baseline error divided by post-lever error, so larger means the
lever helped the attacker more. Brackets are 95% percentile bootstrap
confidence intervals for the mean over seeds (2000 resamples).

## Panel A: the 2x2 taxonomy

| cell | defense | amplification gain | averaging gain | predicted vulnerable to |
|---|---|---:|---:|---|
| absolute + deterministic | rounding (1 dec.) | 301.8x [251.0, 353.4] | 1.0x [1.0, 1.0] | amplification |
| absolute + stochastic | additive noise (0.1) | 252.0x [205.0, 310.0] | 8.0x [7.0, 9.1] | amplification and averaging |
| relative + deterministic | significant digits (3) | 0.9x [0.6, 1.2] | 1.0x [1.0, 1.0] | neither |
| relative + stochastic | multiplicative noise (0.1%) | 0.8x [0.7, 1.0] | 8.7x [7.6, 9.8] | averaging |
| absolute + deterministic (worst-case) | adversarial displacement (0.05) | 256.3x [210.9, 306.4] | 1.0x [1.0, 1.0] | amplification |

Read across: a cell is breached by exactly the lever its properties expose
it to. Only the relative+deterministic cell resists both, which is the
recommendation the paper makes.

## Panel B: does the absolute-grid breach generalize?

`rank` is the mean rank of the per-class augmented design under
amplification, against the rank required to solve. When it falls short the
attack cannot solve the class at all, so a small gain means the *attack*
broke, not that the defense worked.

| dataset | width | family | amplification gain | rank (amp) | required | fail rate |
|---|---:|---|---:|---:|---:|---:|
| digits | 8 | absolute | 301.8x [251.0, 353.4] | 9.0 | 9 | 0.00 |
| digits | 8 | relative | 0.9x [0.6, 1.2] | 9.0 | 9 | 0.00 |
| digits | 16 | absolute | 189.7x [130.2, 251.6] | 17.0 | 17 | 0.00 |
| digits | 16 | relative | 0.4x [0.2, 0.7] | 17.0 | 17 | 0.00 |
| digits | 32 | absolute | 135.1x [83.8, 182.9] | 33.0 | 33 | 0.00 |
| digits | 32 | relative | 0.3x [0.1, 0.4] | 33.0 | 33 | 0.00 |
| wine | 8 | absolute | 3.2x [0.1, 7.9] | 8.1 | 9 | 0.80 |
| wine | 8 | relative | 0.0x [0.0, 0.0] | 8.1 | 9 | 0.80 |
| wine | 12 | absolute | 0.1x [0.1, 0.1] | 10.1 | 13 | 1.00 |
| wine | 12 | relative | 0.0x [0.0, 0.0] | 10.1 | 13 | 1.00 |

Interpretation: the absolute grid is breached where the amplified design
stays full rank, and appears safe only where it does not. Relative
quantization is never breached in any setting, and its safety does not
depend on which of the two regimes the deployment happens to be in.

## Panel C: how homogeneous is the feature map?

Theorem 6 is stated under exact positive homogeneity. `ratio` is
`||phi(Ax)|| / (A ||phi(x)||)`: it does not stay at 1, so the map is not
exactly homogeneous and the hypothesis as originally written is false.
`defect` is what the proof actually needs -- the relative distance between
`phi(Ax)/A` at consecutive scales. It tends to zero, so `phi(Ax)/A`
converges to a fixed matrix and the theorem holds asymptotically, which is
how the paper now states it.

| dataset | quantity | A=1 | A=4 | A=16 | A=64 | A=256 | A=1024 | A=4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| digits | ratio | 1.000 | 1.484 | 1.670 | 1.718 | 1.731 | 1.734 | 1.734 |
| digits | defect | 0.5602 | 0.1400 | 0.0342 | 0.0085 | 0.0021 | 0.0005 | 0.0001 |
| wine | ratio | 1.000 | 6.911 | 8.587 | 9.008 | 9.113 | 9.139 | 9.146 |
| wine | defect | 0.9101 | 0.1974 | 0.0471 | 0.0116 | 0.0029 | 0.0007 | 0.0002 |
