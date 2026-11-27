# Offline Verification Report: IJCAI 2027 Three-Theorem Package

## Theorem 1 & Corollary 1 (3-Outcome Three-Regime EVOI + Rework Shift)
- Boundaries ($p_e=0.75$): $Q_1^* = 5.324$, $Q_2^* = 13.389$, Rework $Q_1^{*,rw} = -1.047$ ($\Delta_{rw} = 5.415$).
- **Frictionless (p_no=0.00)**: R1 slope `0.4675` (theo `0.4675`), R2 slope `-0.3825` (theo `-0.3825`), R3 slope `-0.0` (theo `0.0`)
- **With Friction (p_no=0.15)**: R1 slope `0.425` (theo `0.425`), R2 slope `-0.2975` (theo `-0.2975`), R3 slope `-0.0` (theo `0.0`)

## Theorem 2 & Corollary 2 (GLM-4.5-Air Shrinkage & Miscalibration Bound)
- **k=2**: `n_cal=59` Cap=1/3 `3.234±0.239` ($\epsilon_p=0.1342, \epsilon_\pi=0.2008$) -> `n_cal=139` Cap=1/3 `3.028±0.465` ($\epsilon_p=0.0618, \epsilon_\pi=0.1183$)
- **k=5**: `n_cal=59` Cap=1/3 `3.222±0.262` ($\epsilon_p=0.1252, \epsilon_\pi=0.1994$) -> `n_cal=139` Cap=1/3 `2.998±0.471` ($\epsilon_p=0.061, \epsilon_\pi=0.1079$)
- **k=10**: `n_cal=59` Cap=1/3 `3.199±0.228` ($\epsilon_p=0.1184, \epsilon_\pi=0.2056$) -> `n_cal=139` Cap=1/3 `3.023±0.408` ($\epsilon_p=0.0624, \epsilon_\pi=0.1025$)
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
