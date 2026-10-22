# The Relative Floor Between Powers of the Base

Design {'n': 24, 'h': 6, 'm': 3, 'rank': 2}, seed 0, 3 significant digits against a 3-decimal absolute grid.

## On the subsequence A = 10^k

- relative grid: spread over k = 0..6 is **1.872e-12** (exact invariance is 0)
- absolute grid: F falls from 1.542e-05 to 2.828e-12 across six decades, i.e. like 1/A

## Off the subsequence

- relative grid dips to **1.641e-02x** its subsequence value and rises to 4.832e+00x
- so the limsup is the subsequence value, and whether the liminf is bounded away from zero is settled by the refinement below

- absolute grid stays within [1.044e-08, 1.909e-05] and never returns to its value at A = 1, which is the contrast the security claim needs

## On the subsequence, under *asymptotic* homogeneity

The sweeps above scale the design exactly. Theorem 6(ii) also claims
F(AQ) = F(Q)(1+o(1)) under the asymptotic homogeneity Theorem 5 assumes.
Same design, same direction, z = A*z0 - mu:

| A | F, exact homogeneity | F, asymptotic |
| --- | --- | --- |
| 1e+00 | 2.056161e-05 | 2.497774e-05 |
| 1e+01 | 2.056161e-05 | 4.018671e-05 |
| 1e+02 | 2.056161e-05 | 2.937758e-05 |
| 1e+03 | 2.056161e-05 | 1.092478e-05 |
| 1e+04 | 2.056161e-05 | 3.364726e-05 |
| 1e+05 | 2.056161e-05 | 2.427491e-05 |
| 1e+06 | 2.056161e-05 | 2.093294e-05 |
| 1e+07 | 2.056161e-05 | 2.059875e-05 |
| 1e+08 | 2.056161e-05 | 2.056533e-05 |

- exact-homogeneity spread: **1.872e-12** (invariant, as part (ii) says)
- asymptotic: F(Q) = 2.497774e-05 but the sequence tends to F_inf = 2.056533e-05, a factor **0.8233**
- and F_inf agrees with the floor at Z_inf to 1.81e-04, so the drift is the hypothesis and not the arithmetic

So F does not decay -- the claim that carries the security argument --
but the limit is not F(Q), and the theorem should not say that it is.

## Refining the sweep (nested grids, so the minimum can only fall)

| sweep points | smallest F, as a fraction of the subsequence value |
| --- | --- |
| 1001 | 2.690e-02 |
| 2001 | 2.200e-02 |
| 4001 | 1.641e-02 |
| 8001 | 4.854e-03 |
| 16001 | 4.498e-03 |
| 32001 | 4.498e-03 |
| 64001 | 9.138e-04 |
| 128001 | 9.138e-04 |
| 256001 | 9.138e-04 |

Log-log slope -0.70: the minimum is still falling after a 256-fold refinement, with no sign of a positive barrier. It falls slowly because collapsing F needs two scores near opposite cell edges at once, which is a codimension-two coincidence in a one-parameter family -- so the floor is unprotected off the subsequence without being fragile at a typical amplification.

## Theorem 5(1): the same defect, in the least-squares error

- across A = 1 to 1e6 the error stays in [1.185e-04, 1.989e-04]: no decay, which is the claim that matters
- but across a single decade it ranges over [8.258e-05, 3.579e-04], a factor of **4.33**, so it does not converge to a limit either

