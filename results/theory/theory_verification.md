# Offline Verification Report: IJCAI 2027 Three-Theorem Package

## Theorem 1 & Corollary 1 (3-Outcome Three-Regime EVOI + Rework Shift)
- Boundaries ($p_e=0.75$): $Q_1^* = 5.324$, $Q_2^* = 13.389$, Rework $Q_1^{*,rw} = -1.047$ ($\Delta_{rw} = 5.415$).
- **Frictionless (p_no=0.00)**: R1 slope `0.4675` (theo `0.4675`), R2 slope `-0.3825` (theo `-0.3825`), R3 slope `-0.0` (theo `0.0`)
- **With Friction (p_no=0.15)**: R1 slope `0.425` (theo `0.425`), R2 slope `-0.2975` (theo `-0.2975`), R3 slope `-0.0` (theo `0.0`)

## Theorem 2 & Corollary 2 (GLM-4.5-Air Shrinkage & Miscalibration Bound)
- **k=2** (fixed $N_{test}=180$): `n_cal=60` Cap=1/3 `2.786±0.524` ($\epsilon_p=0.6261, \epsilon_\pi=0.3046$) -> `n_cal=240` Cap=1/3 `3.099±0.589` ($\epsilon_p=0.5248, \epsilon_\pi=0.1907$)
- **k=5** (fixed $N_{test}=180$): `n_cal=60` Cap=1/3 `2.827±0.498` ($\epsilon_p=0.6044, \epsilon_\pi=0.2824$) -> `n_cal=240` Cap=1/3 `3.112±0.606` ($\epsilon_p=0.538, \epsilon_\pi=0.1881$)
- **k=10** (fixed $N_{test}=180$): `n_cal=60` Cap=1/3 `2.96±0.635` ($\epsilon_p=0.5997, \epsilon_\pi=0.2672$) -> `n_cal=240` Cap=1/3 `3.14±0.638` ($\epsilon_p=0.5525, \epsilon_\pi=0.1868$)
- **Qwen3-8B F1 Floor Boundary**: Cap=2 `3.731`, Cap=1/3 `1.714`.

## Theorem 3 (Foster-Lyapunov Stability Expansion & Non-Zero ASK Latency)
- Stability Thresholds ($\mu=0.50$): `{ACT, DEFER}` $\lambda_{base}^* = 0.667$ vs `Tri-VoI-L2D` $\lambda_{Tri}^* = 1.333$ (2.0x capacity expansion).
- Latency Tolerance ($q_0=2$, $\bar{\delta}=0.35$, $\beta_{wait}=0.25$): Closed-form $\tau_{ask}^{\max} = 1.38$ slots.
| $\lambda_{tot}$ | `{ACT, DEFER}` Mean Q | `Tri-VoI-L2D` Mean Q | Lyapunov Bound $B/\varepsilon$ |
|---:|---:|---:|---:|
| 0.30 | 0.51 | 0.16 | 0.66 |
| 0.45 | 1.32 | 0.29 | 0.8 |
| 0.60 | 6.14 | 0.48 | 1.0 |
| 0.75 | 120.01 | 0.76 | 1.32 |
| 0.90 | 361.83 | 1.38 | 1.89 |
| 1.05 | 587.34 | 2.45 | 3.08 |
| 1.20 | 797.11 | 6.35 | 7.02 |
| 1.45 | 1185.86 | 96.96 | Unstable |

### Latency Sweep ($\tau_{ask}$)
| $\tau_{ask}$ | MC EVOI | Regime-I Linear Prediction |
|---:|---:|---:|
| 0 | 0.550 | 0.550 |
| 1 | 0.141 | 0.151 |
| 2 | -0.262 | -0.248 |
| 4 | -1.053 | -1.045 |
| 6 | -1.848 | -1.843 |
| 8 | -2.660 | -2.640 |

### Real `simulate_stream` Empirical Queue vs. Accuracy (`v1-Clean`, `Qwen3.5-35B-A3B`)
| Service Rate $\mu$ | `Congested-L2D-Only` $\bar{Q}$ (Acc%, Util) | `Tri-VoI-L2D` $\bar{Q}$ (Acc%, Util) | `Tri-VoI-MDP` $\bar{Q}$ (Acc%, Util) |
|---:|---:|---:|---:|
| 2.00 | 0.00 (80.72%, 4.870) | 0.00 (83.83%, 5.367) | 0.00 (84.28%, 5.429) |
| 1.00 | 0.32 (80.44%, 4.646) | 0.28 (83.67%, 5.220) | 0.25 (84.11%, 5.341) |
| 0.50 | 5.21 (77.06%, 2.321) | 0.99 (81.50%, 4.610) | 0.67 (81.61%, 4.781) |
| 0.33 | 8.19 (71.83%, 1.447) | 3.38 (79.56%, 3.792) | 2.37 (79.33%, 4.048) |

## Theorem 4 (Exact 3-Action Periodic-Drain Bellman Optimality & Shadow-Price Monotonicity)
- Phase-1 shadow prices $\Delta V_{25}(1, q)$ for $q=0..9$: `[0.4911, 0.8071, 1.4934, 1.9007, 2.4293, 2.9016, 3.4, 3.8862, 4.2466, 4.4452]` (sub-capacity $q=1 < C$: `0.8071` > 0).
- Non-negativity min $\Delta V$: `0.0`, $\gamma=0$ max $|\Delta V|$: `0.0`, Coupling bound ($h=4$): `1.3401 <= 3.4`.
- 4-branch abandonment threshold: $Q_{act}=13.557$, $Q_{\max}^{\text{defer}}=16.919$ (witness $q=14$ gap $\Delta V=0.5894>0$; above 4-branch bound $q=69$: `0.0`), Operating-queue diff ($q\le 7$): `9e-05`.

## Proposition 5 (Tri-Party EVOI Regret Decomposition under Matched Dynamic Shadow Pricing)
| $c_{\text{ask}}$ | $Q$ | $\widetilde{U}_{\text{def,eff}}$ | $\text{EVOI}_{\text{DP}}$ | $\text{Regret}_{\text{CC}}$ | $\Delta_{\text{opt}}$ | $\text{Regret}_{\text{JG}}$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0 | 2.452 | +2.078 | 0.000 | 4.917 | 2.078 |
| 0.5 | 3 | -4.204 | +4.279 | 0.000 | 1.896 | 1.896 |
| 0.5 | 7 | -6.564 | +3.208 | 0.000 | 0.825 | 0.825 |
| 0.5 | 12 | -8.893 | +2.383 | 0.000 | 0.000 | 0.000 |
| 1.5 | 0 | 2.148 | +1.244 | 0.000 | 4.779 | 1.244 |
| 1.5 | 3 | -4.253 | +3.257 | 0.000 | 1.874 | 1.874 |
| 1.5 | 7 | -6.564 | +2.208 | 0.000 | 0.825 | 0.825 |
| 1.5 | 12 | -8.893 | +1.383 | 0.000 | 0.000 | 0.000 |
| 3.0 | 0 | 0.943 | +0.403 | 0.000 | 4.232 | 0.403 |
| 3.0 | 3 | -4.625 | +1.588 | 0.000 | 1.705 | 1.588 |
| 3.0 | 7 | -6.601 | +0.691 | 0.000 | 0.808 | 0.691 |
| 3.0 | 12 | -8.817 | -0.117 | 0.117 | 0.000 | 0.000 |
| 4.5 | 0 | -0.584 | -0.264 | 0.264 | 3.539 | 0.000 |
| 4.5 | 3 | -5.169 | -0.159 | 0.159 | 1.458 | 0.000 |
| 4.5 | 7 | -6.529 | -0.776 | 0.776 | 0.840 | 0.000 |
| 4.5 | 12 | -8.328 | -1.593 | 1.593 | 0.024 | 0.000 |
