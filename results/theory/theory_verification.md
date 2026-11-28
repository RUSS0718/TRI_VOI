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
- Non-negativity min $\Delta V$: `0.0`, Coupling bound ($h=4$): `1.3401 <= 3.4`, Abandonment limit ($q=65 > Q_{\max}+25C$): `0.0`, Operating-queue truncation diff ($q\le 7$, $H=25$ vs $100$): `9e-05`.
