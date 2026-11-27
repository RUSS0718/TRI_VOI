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

### Motivation 3：专家拥塞/漂移与用户澄清的数学耦合——“队列减压阀”效应（Queue-Relief VoI Coupling）
常规看法认为“拥塞与能力漂移仅影响 `DEFER`”。我们指出：**在存在混合不确定性（Conditional Escalation，即澄清用户条件后，部分案例回归常规规则可由 AI 自主解决，另一部分才真正触发内部特例需转专家）的协作任务中，专家资源的动态变化会直接改变向用户发起 `ASK` 的期望信息价值（EVOI）**。

设澄清后以概率 $\hat{p}_{\text{resolve}}$ 落入常规分支（AI 以期望效用 $U_{\text{easy}} = \hat{p}_{\text{easy}} R_+ + (1-\hat{p}_{\text{easy}}) R_-$ 自主完结），以概率 $1-\hat{p}_{\text{resolve}}$ 确认触发特例需转接专家（专家实时效用 $U_{\text{defer}}(Q_t, \hat{p}_e) = \hat{p}_e R_+ + (1-\hat{p}_e) R_- - C_{\text{defer}} - \gamma Q_t$）。对比“不问用户直接兜底转专家”与“先问用户再按结果决策”的两阶段 Bellman 收益差，可得**单调耦合性质（Monotone Coupling Proposition）**：
$$\frac{\partial\,\text{EVOI}(\text{ASK}; Q_t, \hat{p}_e)}{\partial Q_t} = \hat{p}_{\text{resolve}}\gamma > 0, \qquad \frac{\partial\,\text{EVOI}(\text{ASK}; Q_t, \hat{p}_e)}{\partial \hat{p}_e} = -\hat{p}_{\text{resolve}}(R_+ - R_-) < 0$$
**物理含义**：当专家队列长度 $Q_t$ 上升或专家实时可靠性 $\hat{p}_e$ 下降时，`ASK` 不仅降低 AI 自身歧义，更发挥了**前置队列减压阀（Queue-Relief Valve）**的作用——筛除本不必占用稀缺专家的案例，因此其边际价值严格递增。结合一阶 Naor 排队外部性影子成本 $\gamma(Q_t + \hat{\rho}_t(Q_t + 1))$，系统在高拥塞下能主动保护后续工单排队延迟。

---

## 4. 三模块精简方法架构（`Tri-VoI-L2D`）

为避免“算法拼盘”并保持闭式可解释性，方法收敛为 **3 个紧耦合模块**：
1. **信息源解耦诊断与分层经验贝叶斯留出校准（Disentangled Source Diagnosis + Hierarchical Empirical Bayes Held-Out Calibration）**：通过一次结构化槽位核查提取可观测诊断向量 $s = (\text{missing\_user\_info}, \text{potential\_override}, H(x))$，仅在**留出校准集（Held-Out Calibration Split）**上估计无泄漏条件准确率 $\hat{p}_{\text{self}}(s)$ 与互斥三分支澄清后验 $(p_{\text{resolve}}, p_{\text{ovr\_after}}, p_{\text{no\_info}})$（当无用户缺项时执行 `(0, 0, 1)` 信息源门控）。
2. **资源耦合两阶段 Bellman EVOI 控制器与排队外部性定价（Resource-Coupled Two-Stage Bellman Controller with Queue Externality）**：联合评估 $U(\text{ACT})$、$U(\text{DEFER} \mid Q_t, \hat{p}_e, \hat{\rho}_t)$ 与两阶段期望效用 $U(\text{ASK} \mid Q_t, \hat{p}_e, \hat{\rho}_t)$；若执行 `ASK`，融合用户回复重算后验状态，再在 `ACT` 与 `DEFER` 间作最终无偏裁决。
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
3. **无泄漏分层校准 + 5 跨家族大模型 + 严格配对检验**：全面替换人工先验常数，并以 5 个异构 LLM 家族满配 `60 cal + 360 test` 收官。

---

## 6. 主实验结果（Phase 2 `N=360`，`results/report.md` & `results/figs/`）

- **汇总文件**：`results/report.md` · `results/all_metrics.json` · `results/latex_tables.tex`
- **高清论文插图（`.png` + `.pdf`）**：
  - `results/figs/fig1_entropy_collapse.pdf`
  - `results/figs/fig2_queue_relief_coupling.pdf`（三联图：固定态斜率/负对照与 $p_e=0.92$ 阈值翻折 + 动态负载与 Naor 外部性等待压缩 + $c_{\text{ask}}$ 敏感性）
  - `results/figs/fig3_drift_and_multiexpert.pdf`
  - `results/figs/fig4_ablation_and_crossmodel.pdf`

### 6.1 Exp 1：主对比与统计检验（`Qwen/Qwen3.5-35B-A3B`，`N=360` × 5 种子 + 6 条独立 60 题流）

> 统计口径：准确率报告 **McNemar 精确检验**；单连续流效用报告 **循环移动块 Paired Bootstrap（`block=10`, 5,000 次）** 95% CI 与双侧 $p$；6 独立流额外报告 **流级配对 $t$ 检验** 与 **分层块 Bootstrap**；跨 7 个非 Oracle 基线作 **Holm-Bonferroni 多重比较校正**。

| Method | Acc% (1.5) | Mean Util (1.5) | ASK% (1.5) | $\Delta\text{Util}_{1.5}$ [95% CI] | Boot $p_{\text{Holm}}$ (1.5) | McNemar $p_{\text{Holm}}$ (1.5) | Mean Util (2.5) | $\Delta\text{Util}_{2.5}$ [95% CI] | Boot $p_{\text{Holm}}$ (2.5) |
| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: | ---: |
| `Direct-ACT` | 60.00 | 2.000 | 0.0 | +3.030 `[+2.16, +3.94]` | **0.0014** | **1.18e-11** | 2.000 | +2.985 `[+2.13, +3.87]` | **0.0014** |
| `Entropy-Clarify` (`ADEPT`) | 59.72 | 1.674 | 18.1 | +3.356 `[+2.46, +4.25]` | **0.0014** | **1.86e-12** | 1.493 | +3.492 `[+2.59, +4.41]` | **0.0014** |
| `Consensus-Defer` (`DREAM`) | 65.28 | 2.683 | 0.0 | +2.347 `[+1.58, +3.18]` | **0.0014** | **5.55e-07** | 2.683 | +2.303 `[+1.61, +3.01]` | **0.0014** |
| `Congested-L2D-Only` | 80.44 | 4.646 | 0.0 | +0.384 `[-0.22, +1.01]` | 0.6743 | 1.0000 | 4.646 | +0.340 `[-0.17, +0.88]` | 0.3975 |
| `Naive-Cascade` | 69.83 | 3.120 | 18.1 | +1.910 `[+1.23, +2.62]` | **0.0014** | **6.25e-04** | 2.939 | +2.046 `[+1.40, +2.72]` | **0.0014** |
| `Joint-Greedy` | 82.67 | 5.073 | 18.0 | -0.043 `[-0.42, +0.34]` | 0.8246 | 1.0000 | 5.055 | -0.069 `[-0.37, +0.22]` | 0.6561 |
| `Calibrated-Cascade` | 81.83 | 4.881 | 39.7 | +0.149 `[-0.14, +0.45]` | 0.6743 | 1.0000 | 4.484 | **+0.502** `[+0.09, +0.89]` | **0.0426** |
| **`Tri-VoI-L2D` (Ours)** | **82.50** | **5.030** | 31.4 | — | — | — | **4.985** | — | — |
| `Oracle` | 82.44 | 5.120 | 37.5 | -0.090 `[-0.49, +0.33]` | 0.6621 | 0.5190 | 4.745 | +0.241 `[-0.23, +0.72]` | 0.3235 |

- **6 条独立 60 题协作流**：在 `c_ask=1.5` 与 `2.5` 下，对 `Direct-ACT / ADEPT / DREAM / Naive-Cascade` 实现 **6/6 流全胜**（流级配对 $t$ 检验 $p_{\text{Holm}} \le 0.0086$，分层块 Bootstrap $p_{\text{Holm}} = 0.0014$）；在 `c_ask=2.5` 下对 `Calibrated-Cascade` 取得 **5/6 流胜出**（`+0.532 [+0.10, +1.02]`）。

### 6.2 Exp 2：队列减压阀、Naor 外部性、2D 理论相变与 $c_{\text{ask}}$ 扫描
- **高/极端拥塞区反超**：随 `Cap=4 → 2 → 1 → 1/3`，`Q4 ASK` 单调抬升 **`48.2% → 58.4% → 71.8% → 80.0%`**（静态对照 `Decoupled-Nominal-EVOI` 僵化在 `83.3%`）。在 `High (Cap=1)` 下 `Tri-VoI-L2D` 均效用 **`4.743`**（Wait `0.71` vs `Joint-Greedy` 3.986 / `Calibrated-Cascade` 4.618 / `w/o Ext` 4.613）；在 `Severe (Cap=1/3)` 下 **`3.764`**（Wait `3.10` vs `Calibrated-Cascade` 3.168 / `w/o Ext` 2.905 / `Joint-Greedy` 2.235）。
- **2D 固定态扫描（$Q\in\{0..6\}\times p_e\in\{0.75, 0.85, 0.92\}\times p_{\text{resolve}}$）**：`p_resolve=0.00` 负对照斜率严格 **`0.0000`**；`p_resolve ∈ {0.25, 0.50, 0.75, 1.00}` 实测斜率 **`0.2125, 0.4250, 0.6375, 0.8500`**，四位小数吻合理论 $p_{\text{resolve}}\gamma$；在 `p_e=0.92` 时校准 `Q4 ASK` 在 $Q=0\to 1$ 呈现 **`0.0% → 83.3%`** 锐利相变。
- **$c_{\text{ask}}$ 敏感性**：在 $c_{\text{ask}}=3.0 / 5.0$ 下 `Calibrated-Cascade` 崩塌至 `4.285 / 3.491`，而 `Tri-VoI-L2D` 保持 **`4.978 / 4.821`**。

### 6.3 Exp 4：组件消融（`N=360`）
移除解耦诊断（`w/o Disentangled-Diag`）准确率跌 **-18.39 pp**（82.50% → 64.11%）、效用跌 **-2.594**；移除分层留出校准（`w/o Held-Out-Calib`）效用从 `5.030` 降至 `4.843`（`2.5` 下 `4.985 → 4.781`）；移除遗忘因子（`w/o Discounted-Forgetting`）降至 `4.945`（`2.5` 下 `4.860`）；静态 `Decoupled-Nominal-EVOI` 降至 `4.870`（`2.5` 下 `4.473`）；`w/o Queue-Externality` 在极轻载 `Cap=2` 略高（`5.155`）但在 `Cap=1/3` 重拥塞骤降 `-0.859`（`3.764 → 2.905`）。

---

## 7. 5 阶段完整扩展实验成果与论文边界总结（`results/overnight/`）

- **总控汇总**：`results/overnight/OVERNIGHT_MASTER_REPORT.md` · `results/overnight/overnight_latex_tables.tex`
- **新增高清插图**：`fig5_multimodel_5families.pdf` · `fig6_noisy_longform_benchmark.pdf` · `fig7_entropy_kt_stress.pdf` · `fig8_2d_phase_heatmap.pdf` · `fig9_cv_lodo_bursty.pdf`

### 7.1 Stage 1 & 2：5 大家族模型满配 `N=360`（每模型 `60 cal + 360 test`）
| Backbone (`N=360`) | Missing F1 | Ent Collapse % | `Naive-Casc` (1.5) | `Joint-Greedy` (1.5) | `Calib-Casc` (1.5) | **`Tri-VoI` (1.5)** | `Calib-Casc` (2.5) | **`Tri-VoI` (2.5)** | Severe `Cap=1/3`: `Tri` vs `Calib` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `Qwen/Qwen3.5-35B-A3B` | 0.8483 | 77.78 | 3.120 | 5.073 | 4.881 | **5.030** | 4.484 | **4.985** | **3.764** vs 3.168 |
| `deepseek-ai/DeepSeek-V3.2` | 0.8946 | 96.11 | 3.330 | 5.391 | 5.550 | **5.514** | 5.075 | **5.238** | **4.086** vs 3.548 |
| `zai-org/GLM-4.5-Air` | 0.8883 | 94.44 | 2.680 | 3.776 | 4.062 | **4.333** | 3.467 | **3.906** | 2.421 vs 2.802 |
| `Qwen/Qwen3-8B` | 0.2157 | 87.22 | 2.937 | 3.589 | 3.616 | **3.742** | 3.549 | **3.742** | **1.714** vs 0.368 |
| `tencent/Hy4-preview` | 0.9831 | 95.00 | 3.441 | 5.390 | 5.833 | **5.859** | 5.350 | **5.411** | **4.476** vs 3.927 |

### 7.2 Stage 3：420 题多段落职场噪音基准 + 非完美 LLM 用户（`N=360`）
- `Qwen/Qwen3.5-35B-A3B`：`Missing F1 = 0.9364`；`Tri-VoI-L2D` = **`3.553`**（`c_ask=1.5`, `Boot p_Holm = 0.0048` vs `Naive` 2.343）与 **`3.544`**（`c_ask=2.5` vs `Calibrated-Cascade` 2.948），**6 条独立流 6/6 全胜**；
- `tencent/Hy4-preview`：`Missing F1 = 0.9972`；`Tri-VoI-L2D` = **`4.397`**（`1.5`, `Boot p_Holm = 0.0014` vs `Naive` 2.666）与 **`4.104`**（`2.5` vs `Calibrated-Cascade` 3.803）。

### 7.3 Stage 4：熵坍缩 $K \in \{3, 5, 10\} \times T \in \{0.7, 1.0\}$ 压测（120 欠定工单）
哪怕提升至 **10 倍采样 + $T=1.0$**：
- `Qwen/Qwen3.5-35B-A3B` 仍有 **`37.50%`**（`T=0.7` 为 **`59.17%`**）彻底坍缩到 $H=0$（多数投票准确率仅 `53.33%`）；
- `tencent/Hy4-preview` 仍有 **`76.67%`**（`T=0.7` 为 **`90.00%`**）彻底坍缩到 $H=0$（多数投票准确率 `56.67%`），而单次信息源诊断 Missing-F1 达 `0.8483 / 0.9831`。

### 7.4 Stage 5：系统相图、CV、跨域迁移与边界条件
- **56 点 2D 相图（`7 档服务率 μ × 8 档 c_ask`）**：`Tri-VoI-L2D` 优于或持平 `Calibrated-Cascade` **`51/56`** 格，优于或持平 `Joint-Greedy` **`38/56`** 格。
- **5 折分层交叉验证（每模型 420 题）**：`Hy4-preview` `5.918±0.486`、`Qwen3.5` `5.564±0.551`、`DeepSeek-V3.2` `5.522±0.715`、`GLM-4.5-Air` `5.520±0.381`、`Qwen3-8B` `4.767±0.479`；LODO 零样本跨域迁移及突发泊松流量波峰（`4.629` 第一）、三级专家轮班池（`5.426`，最低等待 `0.13`）均保持稳健。
- **客观边界**：(1) 轻载低打扰单点（`Cap=2, c_ask=1.5`）下同诊断强基线单点接近，优势体现在二维工作区全局稳健性；(2) `8B` 小底座存在诊断能力下界（`Missing F1 = 0.2157`）；(3) Stage 3 已引入非完美 LLM 用户与职场噪音文本，尚未开展真人用户受控实验。
