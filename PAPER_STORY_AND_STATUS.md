# Tri-VoI-L2D：面向三方人机协同决策的解耦信息价值与资源耦合延迟控制

> 本文档完整归档 **`Tri-VoI-L2D`** 的论文英文拟定题目、核心研究命题、三大联动理论动机、三模块精简方法、Pilot/MVE 审计及全套 5 阶段正式实证结果（共 **26,039 次**真实 LLM API 调用，全部指标与图表位于 `results/`）。

---

## 1. 拟定题目与适用领域

- **拟定英文题目**：  
  *Ask the User or Defer to the Expert? Information-Source Disentanglement and Queue-Relief Deferral for Three-Way Human-AI Cooperative Decision-Making*
- **核心研究方向**：  
  *Human-AI Cooperative Decision-Making · Value of Information (VoI) · Learning to Defer (L2D) · Group Decision Support*（落地覆盖 **Clinical Triage 临床分诊、Financial Compliance 金融合规、IT Incident Dispatch 运维工单、Logistics SLA 物流调度** 四大协同决策域）。

---

## 2. 核心研究命题（Unified Research Thesis）

> **有效的人机协作不仅取决于 AI 对自身不确定性的感知，更取决于识别“谁持有消除该不确定性所需的信息”（一线请求者 vs. 内部领域专家），并将向用户澄清的价值与动态受限的专家队列资源耦合决策。**

围绕这一命题，全文由三个逐层递进的科学问题（Research Questions）贯穿：
1. **RQ1（Who to ask?）**：解耦不确定性来源（用户侧规范缺失 vs. 专家侧政策/特例盲区）与无泄漏留出校准，能否解决真实 LLM 的高共识盲猜坍缩并准确路由到正确的协同对象？
2. **RQ2（When to ask?）**：在真实自然语言协作工单中，显式建模“澄清后收益”的两阶段 Bellman EVOI 门控，能否在准确率与交互/人工总效用帕累托前沿上显著超越单向澄清与单向延迟方法？
3. **RQ3（How to adapt?）**：当专家队列拥塞加剧或发生突发/渐进疲劳漂移（以及异质 Senior/Mid/Junior 专家协作、突发泊松流量波峰、非对称高风险惩罚）时，资源耦合策略如何通过动态提升 `ASK` 的**队列减压阀（Queue-Relief Valve）**价值保持协同系统稳健？

---

## 3. 理论动机（Three Linked Motivations & Coupling Property）

### Motivation 1：现有协同框架的“二元路由割裂”（Binary Routing Fallacy）
真实协作工单流处于**三方协作拓扑**（Requester 用户 ↔ AI Agent ↔ Constrained Specialist Pool 人类专家池），同时面临两类正交不确定性：
- **规范/情境歧义（Specification Uncertainty, $U_{\text{spec}}$）**：由用户持有的私有信息缺失引起，**只有用户知道**（若误转专家，专家仍无法直接裁决，徒增往返与队列占用）。
- **政策/特例认知不确定（Epistemic / Policy Uncertainty, $U_{\text{epist}}$）**：由未公开内部例外条款、复杂边界规约引起，**只有领域专家知道**（若误问用户，用户无法回答，徒增交互疲劳）。

近期顶会工作被人为割裂为两条互不连通的单边路线：
- **澄清派（`{ACT, ASK}`）**：如 `ADEPT`、`VoI for Human-Agent Communication`，不具备向专家升级的通道；
- **延迟派（`{ACT, DEFER}`）**：如 `DREAM`、`Online Learning with LLMs and Congested Experts`、`Online L2D with Varying Experts`，假定用户输入完整无歧义，遇到不确定只能直接转接专家。

### Motivation 2：真实 LLM 的“欠定高共识熵坍缩”（High-Consensus Entropy Collapse）
在没有显式占位符剧透的真实工单中，当某个决定性条件被自然遗漏时，现代指令微调 LLM 在随机采样下往往高度一致地默认补全某个常见先验假设（香农熵 $H=0$、一致性 3/3），表现为“**高度一致地盲猜**”。这直接导致依赖采样熵（`ADEPT`）或自洽分歧（`DREAM`）的触发器大面积漏检欠定案例——哪怕提高采样数到 $K=10$ 且升温到 $T=1.0$，这一现象依然顽固存在。

### Motivation 3：专家拥塞/漂移与用户澄清的三区间分段耦合——“队列减压阀”定理（Three-Regime Piecewise Queue-Relief VoI）
常规看法认为“拥塞与能力漂移仅影响 `DEFER`”。我们指出：**在存在混合不确定性（Conditional Escalation，即澄清用户条件后，以概率 $\hat{p}_{\text{resolve}}$ 回归常规规则由 AI 自主解决、以概率 $1-\hat{p}_{\text{resolve}}$ 触发特例需转专家）的协作任务中，专家队列与可靠性的动态变化直接重塑向用户发起 `ASK` 的期望信息价值（EVOI）**。

设 $U_{\text{defer}}(Q_t, \hat{p}_e) = \hat{p}_e R_+ + (1-\hat{p}_e) R_- - C_{\text{defer}} - \gamma Q_t$，定义两个闭式队列边界：
$$Q_1^* = \frac{\hat{p}_e R_+ + (1-\hat{p}_e)R_- - C_{\text{defer}} - U_{\text{act}}}{\gamma}, \qquad Q_2^* = \frac{\hat{p}_e R_+ + (1-\hat{p}_e)R_- - C_{\text{defer}} - U_{\text{ovr}}}{\gamma}$$
对比两阶段 Bellman 收益差 $\text{EVOI}(\text{ASK}) = U_{\text{ASK}} - \max(U_{\text{act}}, U_{\text{defer}})$，可得精确的**三区间分段定理（Three-Regime Piecewise Theorem）**：
1. **Regime I（$Q_t < Q_1^*$，`DEFER` 主导无问基线）**：$\frac{\partial\,\text{EVOI}}{\partial Q_t} = \hat{p}_{\text{resolve}}\gamma > 0$，且 $\frac{\partial\,\text{EVOI}}{\partial \hat{p}_e} = -\hat{p}_{\text{resolve}}(R_+ - R_-) < 0$（`ASK` 充当**前置队列减压阀**）；
2. **Regime II（$Q_1^* \le Q_t < Q_2^*$，翻转区间）**：无问基线切向 `ACT`，而确认特例分支仍依赖专家，斜率反转为负值 $-(1-\hat{p}_{\text{resolve}})\gamma < 0$；
3. **Regime III（$Q_t \ge Q_2^*$，饱和区间）**：澄清后即便确认特例也因排队过高而回退 `ACT`，斜率归零 $\frac{\partial\,\text{EVOI}}{\partial Q_t} = 0$。

结合**拥塞门控的 Naor 排队外部性影子成本**（仅当队列结转 $Q_t \ge C_m$ 或高负载 $\hat{\rho}_t \ge 0.6$ 时触发 $\gamma(Q_t + \hat{\rho}_t(Q_t + 1))$，避免轻载误伤），系统兼顾轻载效用与重载排队保护。

---

## 4. 三模块精简方法架构（`Tri-VoI-L2D`）

为避免“算法拼盘”并保持闭式可解释性，方法收敛为 **3 个紧耦合模块**：
1. **信息源解耦诊断与分层经验贝叶斯留出校准（Disentangled Source Diagnosis + Hierarchical Empirical Bayes Held-Out Calibration）**：通过一次结构化槽位核查提取可观测诊断向量 $s = (\text{missing\_user\_info}, \text{potential\_override}, H(x))$，仅在**留出校准集（Held-Out Calibration Split）**上估计无泄漏条件准确率 $\hat{p}_{\text{self}}(s)$ 与互斥三分支澄清后验 $(p_{\text{resolve}}, p_{\text{ovr\_after}}, p_{\text{no\_info}})$（`p_post_clear` 严格仅统计真实发起且成功澄清的校准样本，无 fallback 混入）。
2. **资源耦合两阶段 Bellman EVOI 控制器与拥塞门控排队外部性（Resource-Coupled Two-Stage Bellman Controller with Gated Queue Externality）**：联合评估 $U(\text{ACT})$、$U(\text{DEFER} \mid Q_t, \hat{p}_e, \hat{\rho}_t)$ 与两阶段期望效用 $U(\text{ASK} \mid Q_t, \hat{p}_e, \hat{\rho}_t)$；若执行 `ASK`，融合用户回复重算后验状态，再在 `ACT` 与 `DEFER` 间作最终无偏裁决。
3. **折扣 Beta-Bernoulli 专家状态跟踪器（Discounted Beta-Bernoulli Tracker）**：以 2 行闭式共轭更新 $\alpha_t = \lambda \alpha_{t-1} + y_t,\; \beta_t = \lambda \beta_{t-1} + (1 - y_t)$（遗忘因子 $\lambda = 0.92$）替代易被早期噪声冻结的普通 EMA 与复杂变点检测器，平滑适应突发/渐进疲劳漂移与排队拥塞。

---

## 5. Pilot MVE 审计与正式基准演进

已完成首轮实时 Pilot MVE（SiliconFlow `Qwen/Qwen3.5-35B-A3B`，`enable_thinking=False`，120 题，960 次调用，447,108 tokens）：

| Policy | Acc (%) | Net Utility | Mean Util | ASK (%) | DEFER (%) | Misroute (%) | Mean Wait | Phase2 Drift Util |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Direct-ACT` | 65.00 | 360.0 | 3.000 | 0.0 | 0.0 | 0.0 | 0.00 | 220.0 |
| `Entropy-Clarify` (`ADEPT`-lite) | 64.17 | 304.0 | 2.533 | 20.0 | 0.0 | 9.17 | 0.00 | 163.5 |
| `Consensus-Defer` (`DREAM`-lite) | 68.33 | 390.4 | 3.253 | 0.0 | 20.0 | 6.67 | 0.08 | 197.2 |
| `Naive-Cascade` | 82.50 | 623.6 | 5.197 | 20.0 | 45.8 | 9.17 | 0.24 | **289.5** |
| **`Tri-VoI-L2D` (Pilot)** | **84.17** | **661.0** | **5.508** | 15.0 | 48.3 | 10.00 | 0.34 | 244.3 |

**MVE 暴露并已在正式实验攻克的 3 项核心问题**：
1. **去除 `NOT PROVIDED` 剧透**：正式基准全部改写为无剧透自然语言工单（并在 Stage 3 进一步扩写为多段落职场噪音文本）。
2. **`Q4` 50/50 Conditional Escalation 重构**：半数澄清后可直接由 AI 解决、半数确认触发内部特例，修复 `Q4_escalate` 标签碰撞后清晰分离出队列减压阀效应。
4. **代码卫生与盲点二次审计（2026-11 修复）**：(a) 移除 `controller.py` / `overnight_lib.py` 中 `clarify_q` 为空时的 `missing_concept` fallback 及 `missing/provide/unspecified` 通用关键词（审计 26,039 条缓存确认 `Tri-VoI` 自身 `0` 污染，修复前反而误惠及 `Entropy-ASK`/`Naive-Cascade`）；(b) 将轻载 Naor 触发从 `q_m > 0` 修正为队列结转条件 `q_m >= C_m`；(c) 为强基线补配 `Calibrated-Cascade+Naor` 与 `Joint-Greedy+Naor`，并将简化启发式基线诚实命名为 `Entropy-ASK (Proxy)` / `Consensus-DEFER (Proxy)`；(d) 修正“4 个模型家族的 5 个模型”口径并新增 `benchmark_v2`、决策相关子集（Stage 5e）及延迟反馈/未澄清返工压测（Stage 5f）。

---

## 6. 主实验结果（Phase 2 `N=360`，`results/report.md` & `results/figs/`）

- **汇总文件**：`results/report.md` · `results/all_metrics.json` · `results/latex_tables.tex`
- **高清论文插图（`.png` + `.pdf`）**：
  - `results/figs/fig1_entropy_collapse.pdf`
  - `results/figs/fig2_queue_relief_coupling.pdf`（三联图：固定态斜率/负对照与 $p_e=0.92$ 阈值翻折 + 动态负载与 Naor 外部性等待压缩 + $c_{\text{ask}}$ 敏感性）
  - `results/figs/fig3_drift_and_multiexpert.pdf`
  - `results/figs/fig4_ablation_and_crossmodel.pdf`

### 6.1 Exp 1：主对比与统计检验（`Qwen/Qwen3.5-35B-A3B`，`N=360` × 5 种子 + 6 条独立 60 题流，零泄漏重跑）

> 统计口径：准确率报告 **McNemar 精确检验**；单连续流效用报告 **循环移动块 Paired Bootstrap（`block=10`, 5,000 次）** 95% CI 与双侧 $p$；6 独立流额外报告 **流级配对 $t$ 检验** 与 **分层块 Bootstrap**；跨 9 个非 Oracle 基线作 **Holm-Bonferroni 多重比较校正**。

| Method | Acc% (1.5) | Mean Util (1.5) | ASK% (1.5) | $\Delta\text{Util}_{1.5}$ [95% CI] | Boot $p_{\text{Holm}}$ (1.5) | Mean Util (2.5) | $\Delta\text{Util}_{2.5}$ [95% CI] | Boot $p_{\text{Holm}}$ (2.5) |
| :--- | ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: |
| `Direct-ACT` | 60.00 | 2.000 | 0.0 | +3.220 `[+2.38, +4.09]` | **0.0018** | 2.000 | +3.142 `[+2.28, +4.02]` | **0.0018** |
| `Entropy-ASK (Proxy)` | 61.11 | 1.951 | 18.1 | +3.269 `[+2.38, +4.17]` | **0.0018** | 1.771 | +3.372 `[+2.49, +4.28]` | **0.0018** |
| `Consensus-DEFER (Proxy)` | 65.28 | 2.683 | 0.0 | +2.538 `[+1.83, +3.26]` | **0.0018** | 2.683 | +2.460 `[+1.75, +3.18]` | **0.0018** |
| `Congested-L2D-Only` | 80.44 | 4.646 | 0.0 | +0.575 `[+0.04, +1.13]` | 0.1930 | 4.646 | +0.497 `[+0.05, +0.98]` | 0.1128 |
| `Naive-Cascade` | 70.17 | 3.165 | 18.1 | +2.055 `[+1.35, +2.78]` | **0.0018** | 2.985 | +2.158 `[+1.46, +2.88]` | **0.0018** |
| `Joint-Greedy` | 83.17 | 5.142 | 14.1 | +0.078 `[-0.18, +0.35]` | 1.0000 | 5.063 | +0.079 `[-0.29, +0.43]` | 0.8650 |
| `Joint-Greedy+Naor` | 83.33 | 5.190 | 15.3 | +0.031 `[-0.26, +0.33]` | 1.0000 | 5.015 | +0.127 `[-0.19, +0.44]` | 0.8650 |
| `Calibrated-Cascade` | 82.00 | 4.877 | 39.7 | +0.344 `[-0.05, +0.73]` | 0.3111 | 4.479 | **+0.663** `[+0.27, +1.07]` | **0.0080** |
| `Calibrated-Cascade+Naor` | 81.94 | 4.867 | 39.7 | +0.353 `[-0.05, +0.73]` | 0.3111 | 4.470 | **+0.672** `[+0.28, +1.08]` | **0.0080** |
| **`Tri-VoI-L2D` (Ours)** | **83.67** | **5.220** | 24.7 | — | — | **5.142** | — | — |
| `Oracle` | 82.72 | 5.175 | 37.5 | +0.045 `[-0.33, +0.43]` | 0.8118 | 4.800 | +0.342 `[-0.09, +0.78]` | 0.1208 |

- **6 条独立 60 题协作流**：在 `c_ask=1.5` 与 `2.5` 下，对 `Direct-ACT / Entropy-ASK (Proxy) / Consensus-DEFER (Proxy) / Naive-Cascade` 实现 **6/6 流全胜**（流级配对 $t$ 检验 $p_{\text{Holm}} \le 0.0152$，分层块 Bootstrap $p_{\text{Holm}} = 0.0018$）；在 `c_ask=2.5` 下对 `Calibrated-Cascade` 与 `Calibrated-Cascade+Naor` 均取得 **5/6 流胜出**（`Strat Boot p_Holm = 0.0056 / 0.0050`）。

### 6.2 Exp 2：队列减压阀、公平 Naor 基线对比、三区间理论检验与 $c_{\text{ask}}$ 扫描
- **机制分离（`Queue-Relief VoI` vs `Naor Externality`）**：在 `Severe (Cap=1/3, c=1.5)` 下，`Tri-VoI-L2D`（`3.792`, Wait `3.06`）显著高于 `Joint-Greedy`（`2.212`）与 `Joint-Greedy+Naor`（`3.188`）；同时公平补配 Naor 后 `Calibrated-Cascade+Naor` 升至 `3.798`（Wait `2.98`），而 `w/o Queue-Externality` 降至 `2.732`（Wait `6.66`）。这清晰证明：**极端拥塞排队抑制主要由 Naor 外部性驱动，而高打扰代价（`c_ask >= 2.5`）与二维全工作区稳健性由两阶段 EVOI 门控主导**（`c_ask=5.0` 下 `Calibrated-Cascade+Naor` 跌至 `3.477`，`Tri-VoI-L2D` 保持 `4.821`）。
- **三区间固定态理论 + 真实 Monte Carlo Rollout（$Q\in\{0..10\}$, `N_MC=200`）**：在 $p_e=0.75$（$Q_1^*=5.32$）时，解析 EVOI 于 $Q=6\to 10$ 出现理论预测的**非线性翻转**（`+0.779 -> +2.912 -> +1.316`），真实采样专家成败的 MC rollout 同样复现翻转（`-1.176 -> +1.000 -> +0.327`）；在 `p_e ∈ {0.75, 0.85, 0.92}` 下，Defer 主导区 MC 实测斜率分别为 `0.4568±0.0137`、`0.4411±0.0065`、`0.4426±0.0036`，吻合理论 $\hat{p}_{\text{resolve}}\gamma = 0.4512$（不再采用同公式拟合自证）。

### 6.3 Exp 4：组件消融（`N=360`）
移除解耦诊断（`w/o Disentangled-Diag`）准确率跌 **-17.11 pp**（83.67% → 66.56%）、效用跌 **-2.342**（`5.220 → 2.878`）；移除分层留出校准（`w/o Held-Out-Calib`）效用降至 `4.921`（`2.5` 下 `5.142 → 4.702`）；移除遗忘因子（`w/o Discounted-Forgetting`）降至 `5.098`（`2.5` 下 `4.808`）；静态 `Decoupled-Nominal-EVOI` 降至 `4.867`；修正轻载触发后 `w/o Queue-Externality` 在 `Cap=2` 仅高 `0.054`（`5.274 vs 5.220`），但在 `Cap=1/3` 重拥塞骤降 `-1.060`（`3.792 → 2.732`）。

---

## 7. 5 阶段完整扩展实验成果与客观边界总结（`results/overnight/`）

- **总控汇总**：`results/overnight/OVERNIGHT_MASTER_REPORT.md` · `results/overnight/overnight_latex_tables.tex`
- **高清插图**：`fig5_multimodel_5families.pdf` · `fig6_noisy_longform_benchmark.pdf` · `fig7_entropy_kt_stress.pdf` · `fig8_2d_phase_heatmap.pdf` · `fig9_cv_lodo_bursty.pdf`

### 7.1 Stage 1 & 2：4 个模型家族、5 个模型的满配评测（每模型 `60 cal + 360 test`）
| Backbone (`N=360`) | Missing F1 | Ent Coll% | `Naive-Casc` (1.5) | `J-Greedy` (1.5) | `Calib-Casc` (1.5) | **`Tri-VoI` (1.5)** | `Calib-Casc` (2.5) | **`Tri-VoI` (2.5)** | Severe `Cap=1/3`: `Tri` / `Calib+Naor` / `w/oExt` / `Naive` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `Qwen/Qwen3.5-35B-A3B` | 0.8483 | 77.78 | 3.165 | 5.142 | 4.877 | **5.220** | 4.479 | **5.142** | 3.792 / 3.798 / 2.732 / 3.000 |
| `deepseek-ai/DeepSeek-V3.2` | 0.8946 | 96.11 | 3.305 | 5.391 | 5.550 | **5.523** | 5.075 | **5.259** | 4.062 / 4.080 / 3.470 / 3.185 |
| `zai-org/GLM-4.5-Air` | 0.8883 | 94.44 | 2.597 | 3.750 | 4.054 | **4.317** | 3.459 | **3.910** | 2.348 / 3.284 / 1.966 / 2.540 |
| `Qwen/Qwen3-8B` | 0.2157 | 87.22 | 2.682 | 3.589 | 3.711 | **3.731** | 3.644 | **3.731** | 1.714 / 1.836 / 0.245 / 2.434 |
| `tencent/Hy4-preview` | 0.9831 | 95.00 | 3.441 | 5.390 | 5.833 | **5.842** | 5.350 | **5.468** | 4.477 / 4.548 / 3.846 / 3.349 |

### 7.2 Stage 3：420 题多段落职场噪音基准 + 非完美 LLM 用户（`N=360`）
- `Qwen/Qwen3.5-35B-A3B`：`Missing F1 = 0.9364`；`Tri-VoI-L2D` = **`3.503`**（`c_ask=1.5`, `Boot p_Holm = 0.0096` vs `Naive` 2.285）与 **`3.511`**（`c_ask=2.5` vs `Calibrated-Cascade` 2.948）；`Cap=1/3` 下 `Tri-VoI-L2D` **`1.525`** vs `Joint-Greedy+Naor` `1.295` / `Calibrated-Cascade+Naor` `1.614`；
- `tencent/Hy4-preview`：`Missing F1 = 0.9972`；`Tri-VoI-L2D` = **`4.380`**（`1.5`, `Boot p_Holm = 0.0018` vs `Naive` 2.666）与 **`4.082`**（`2.5` vs `Calibrated-Cascade` 3.803）。

### 7.3 Stage 4：熵坍缩 $K \in \{3, 5, 10\} \times T \in \{0.7, 1.0\}$ 压测（120 欠定工单）
哪怕提升至 **10 倍采样 + $T=1.0$**：
- `Qwen/Qwen3.5-35B-A3B` 仍有 **`37.50%`**（`T=0.7` 为 **`59.17%`**）彻底坍缩到 $H=0$（多数投票准确率仅 `53.33%`）；
- `tencent/Hy4-preview` 仍有 **`76.67%`**（`T=0.7` 为 **`90.00%`**）彻底坍缩到 $H=0$（多数投票准确率 `56.67%`），而单次信息源诊断 Missing-F1 达 `0.8483 / 0.9831`。

### 7.4 Stage 5：2D 相图、决策相关子集（5e）、真实专家压测（5f）与边界条件
- **56 点 2D 相图（`7 档服务率 μ × 8 档 c_ask`, 8 方法）**：`Tri-VoI-L2D` 在 **27/56** 格取得全场最高效用（`Calib-Cascade+Naor` 13/56、`Joint-Greedy` 8/56）；成对优于/持平 `Calibrated-Cascade` **`53/56`**、`Calibrated-Cascade+Naor` **`43/56`**、`Joint-Greedy` **`46/56`**、`Joint-Greedy+Naor` **`44/56`**、`w/o Queue-Externality` **`49/56`**。
- **Stage 5e（剔除 34.3% 缺项不改变金标的 Q2 工单，留存 `328/360` 题）**：结论稳健，`Tri-VoI-L2D` 在 `(Cap=2, c=1.5)` / `(Cap=2, c=2.5)` / `(Cap=1/3, c=1.5)` 分别达到 **`5.253`** / **`5.141`** / **`3.622`**（对比 `Joint-Greedy`: `5.199 / 5.044 / 2.150`，`Calib-Cascade`: `4.941 / 4.557 / 2.858`）。
- **Stage 5f（排队延迟反馈 + 未澄清转专家返工惩罚）**：在 `Cap=1/3` 双重压力下，`Tri-VoI-L2D` 保持 **`3.561`**，显著优于 `Joint-Greedy+Naor`（`2.823`）与 `Calibrated-Cascade`（`2.765`）；但在 `Cap=2` 纯返工场景中，`Tri-VoI-L2D`（`4.279`）因 $U_{\text{defer}}$ 尚未显式计入返工惩罚而低于无脑必问的 `Calibrated-Cascade`（`4.678`），暴露了后续工作可在效用模型中显式建模返工风险的改进空间。
- **客观边界**：(1) 轻载低打扰单点上，与同诊断强基线差异温和，主优势体现于高成本×高拥塞二维工作区；(2) `GLM-4.5-Air` 在 `Cap=1/3` 下受 `n=60` 小校准集 Q2 方差影响输给 `Calibrated-Cascade+Naor`（`2.348 vs 3.284`），`Qwen3-8B` 受 `Missing F1=0.2157` 限制在 `Cap=1/3` 低于 `Naive-Cascade`（`1.714 vs 2.434`）；(3) `build_cscw_benchmark_v2`（100% 决策相关 Q2 + 无 `confidential/override` 显式词）与独立公开数据集评测列为 P1 待完成实证，目标投稿定位 **IJCAI / ECAI / AAAI** AI 决策路线。
