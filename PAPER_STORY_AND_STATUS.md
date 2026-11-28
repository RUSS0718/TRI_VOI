# IJCAI 2027 Paper Story, Contribution Positioning & Verification Ledger

**Primary Target**: IJCAI 2027  
**Unified Primary Method**: **`Tri-VoI-MDP` (`Deploy-Self`, 100% Zero-Oracle)**  
**Single Source of Truth**: [ijcai2027_canonical_tables.md](file:///d:/research/cscwd_tri_voi/results/ijcai2027_canonical_tables.md) & [ijcai2027_three_theorem_proofs.md](file:///d:/research/cscwd_tri_voi/docs/theory/ijcai2027_three_theorem_proofs.md)

---

## 1. 核心定位与方法演进（为什么统一为 `Tri-VoI-MDP`）

为消除早期“方法命名分裂（`L2D` vs `MDP`）”及“Oracle 信号泄露”隐患，当前 `main` 分支已与历史阶段彻底切割（旧版本与过夜消融完整固化于只读归档分支 `archive/v1-tri-voi-l2d`、`archive/v2-mdp-oracle-diag`、`archive/full-history-snapshot`），仓库主叙事严格统一到 **`Tri-VoI-MDP`（零 Oracle 可部署版）**：

| 版本 / 标签 | 探测预算 | 澄清后信息信号 | 队列定价机制 | 在论文中的正式角色 |
|:---|:---|:---|:---|:---|
| **`Tri-VoI-L2D`** | 5-Call | 5-call 观测信号 | 静态 Naor 阈值 | **前代基线（Theorems 1–3）**：奠定三方两阶段序贯探测与次模 EVOI 理论基础；在 `v1-Clean / Qwen3.5` 冻结复现 `[5.220, 5.142, 3.792]`。 |
| **`Tri-VoI-MDP` (`Deploy-Self`)** | **2-Call 粗充分统计量** | **严格本地 LLM `post_missing`**（仅回退至 `obs_missing`，零访问 `oracle_revealed`） | **有限时域周期排空 Bellman DP 影子价格** ($\Delta V(q)$) | **本文唯一主方法（Theorem 4 & Proposition 5）**：面向 IJCAI 2027 全文表格（Sections 0–10）的零泄露、低 Token、高鲁棒主控制器。 |
| `MDP (Oracle-NoInfo Diag)` | 2-Call | 读取模拟器 `not oracle_revealed` | Bellman DP | **仅限消融诊断列（Sections 9A/9B/10）**：用于量化模拟器信息边界，不作为任何主结论性能声明。 |

---

## 2. 论文三大核心贡献与硬核证据（100% 源自 Sections 0–10）

我们**不**主张 `Tri-VoI-MDP` 在每一个孤立单格（single cell）上都统计学显著碾压配对的 `2C-Self Base+MDP` 基线；相反，我们以机制分解、最坏情况极小化极大后悔（Minimax Regret）和临床跨系统盲测构筑更稳固的 A 会叙事：

### 贡献一：三方信息获取中的动态 Bellman 拥塞定价机制（Theorem 4 + Section 0A + Fig. 1a / Fig. 2c）
- **理论**：Theorem 4 在严格限定范围（已知转移参数、单专家服务池、有限时域周期排空、`γ > 0`，且考虑澄清后三分支效用）证明了队列长度 $q$ 上的最优放弃单调阈值结构。
- **机制消融（Section 0A，高拥塞 `Cap=1/3, c=1.5`，13 个评估套件）**：
  只要将静态 Naor 阈值替换为动态 Bellman 影子价格（`+MDP`），跨策略家族呈现普适提升：
  - `CC+Naor -> CC+MDP`：**13/13** 正增益（`min=+0.063, median=+0.280, mean=+0.355, max=+0.804`）
  - `ACL+FD+Naor -> ACL+FD+MDP`：**13/13** 正增益（`min=+0.063, median=+0.447, mean=+0.443, max=+0.943`）
  - `Tri-VoI-L2D -> Tri-VoI-MDP (Zero-Oracle)`：**12/13** 正增益（`min=-0.124, median=+0.435, mean=+0.429, max=+0.914`）
  - `JG+Naor -> JG+MDP`：**12/13** 正增益（`min=-0.029, median=+0.370, mean=+0.429, max=+0.824`）

### 贡献二：低开销（2-Call）条件下的跨场景 Minimax Regret 鲁棒性（Proposition 5 + Sections 0B, 3, 6 + Fig. 1b / Fig. 2d / Fig. 3）
- **Token 成本直降 53.7%–57.8%（Section 6 / Fig. 3a）**：相比 5-Call 完整探测，2-Call `Tri-VoI-MDP` 在 `v1-Clean` 与 `v2` 上将实测 SQLite Token 从 `4373–4967` 降至 `2014–2125`（例如 `v1-Clean / Qwen3.5`：`4952 -> 2111`，**-57.4%**；`v2 / Qwen3.5`：`4967 -> 2097`，**-57.8%**）。
- **严格配对 `2C-Self` 跨网格最坏后悔最小（Section 0B / Fig. 1b）**：
  - 在 `26` 个严格配对单元（`13 套件 × {Cap=2, Cap=1/3}, c=1.5`，同为 2-Call、零 Oracle、同 Bellman 定价）中，`Tri-VoI-MDP` 取得最高平均效用 **3.617**、最多胜场 **10/26**、最低平均后悔 **0.099** 与最低最坏后悔 **0.464**（对比 `CC+MDP (2C-S)`: `3.512 / 0.203 / 0.625`；`JG+MDP (2C-S)`: `3.480 / 0.236 / 0.820`；`ACL+FD+MDP (2C-S)`: `3.472 / 0.243 / 0.764`）。
  - 在 `40` 个澄清代价扫描单元（`c_ask ∈ {0.5, 1.5, 3.0, 4.5}`）中，`Tri-VoI-MDP` 取得最高平均效用 **3.380** 与最低最坏后悔 **0.526**（对比 `JG+MDP (2C-S)`: `0.927`，`CC+MDP (2C-S)`: `1.330`）。
  - 在 56 格 2D 灵敏度包络（`Table 3`）中，`Tri-VoI-MDP` 在 **5/7** 个套件上同时占据最高均值与最低最坏后悔。

### 贡献三：预注册跨器官系统盲测与真实 LLM Specialist 验证（Sections 8 & 10 + Fig. 4）
- **`MediQ-Test-Disjoint`（`prereg-mediq-test-disjoint`，`N_cal=60, N_test=180`，`PY_DIFF_LEN=0`，`Unparsed=0`，花费 `¥3.55`）**：
  按医学器官系统严格零重叠切分校准集（`{Cardio_Pulm, Endo_Renal, Other}`）与测试集（`{Heme_Onc_Repro_MSK, Infectious_Immuno, GI_Hepatic, Neuro_Psych}`），且长度匹配（`Q1` vs `Q2` 文本差异仅 `4.86%`）。
  - 在 **Setup A（`Qwen3.5-35B-A3B -> DeepSeek-V3.2` 真实 LLM Specialist）** 中，`Tri-VoI-MDP (Zero-Oracle)` 在 `Cap=2, c=1.5` 达 **4.266**（原始答题准确率 `81.7%`，越界转交召回 `85.1%`，`1823` tokens），在 `Cap=1/3, c=1.5` 达 **3.588**（`82.2% / 71.6%`），相对 `Tri-VoI-L2D` 显著提升 `+0.679`（`p=0.0137`；`c=3.0` 时 `+0.785, p=0.0057`）；同时高拥塞下所有 4 组 `+MDP` 均显著优于对应 `+Naor`（`p < 0.05`）。
- **零 Oracle 合规性审计（Section 9C）**：全部 `4,974` 条历史 `post_ask_combined*` 记录中，仅 `5` 条因 JSON 截断缺失 `missing_user_info`，主方法均严格回退至 `obs_missing`；新增 `MediQ-Test-Disjoint` 盲测中 JSON 解析失败回退为 `0`。

---

## 3. 诚实边界与审稿披露（Limitations — 必须正文保留）

1. **单格配对 `2C-Self Base+MDP` 非普适显著优胜（Sections 9E & 10B / Fig. 4b）**：
   在同等 2-Call、零 Oracle 与相同 Bellman 定价下，单格 Bootstrap 检验显示 `Tri-VoI-MDP` 与最强 `Base+MDP (2C-S)` 多数处于同一统计水平（`Section 9E` 中 `p=0.2036–0.9810`）；在 `v2 (Qwen3.5)`、`v2 (Hy4-preview)` 及 `MediQ-Test-Disjoint Setup B` 部分单元上，`CC+MDP (2C-S)`、`ACL+FD+MDP (2C-S)` 或 `JG+Naor` 数值领先。论文明确将独立算法价值表述为**最坏情况后悔控制与少调用鲁棒性**，而非逐格全面碾压。
2. **`Benchmark v3` 模板重合与零重叠切片（Sections 8 & 9D）**：
   `v3` 每个未见领域由 `2×2×2=8` 个离散属性模板生成，导致 `72/144` (`50.0%`) 测试样本与校准集存在归一化文本重合、`51/144` (`35.4%`) 存在 `(prose, missing_val, gold)` 三元组重合；已在正文将 `v3` 定性为**模板级结构泛化基准**，并在 `Section 9D` 单列 `Zero-Cal-Prose-Overlap (72)`（`Qwen3.5 Cap=2, c=1.5` 达 **5.559**）与 `Unique-Prose (40)` 切片（与 `2C-S` 呈统计持平 `p=0.1546–1.0000`），真实去重外推结论严格依托 `MediQ-Holdout` 与 `MediQ-Test-Disjoint`。
3. **`post_missing` 自监督信号质量与越界转交权衡（Sections 7b, 9C）**：
   `post_missing` 在高噪声场景识别有效（`v1-Noisy` `F1=0.857`），但在低失效样本数据集上稀疏（`v1-Clean` `6` 例仅召回 `0`，`MediQ-Holdout` `F1=0.267`）；此外在严重拥塞（`Cap=1/3`）下，理性队列定价会伴随 `EscRec` 从 `~81–90%` 降至 `55.6%–71.6%`，并在弱全科模型基座（`Qwen3-8B`）出现探测失灵边际。

---

## 4. 论文图表资产与分支归档索引

- 主结果总表：[results/ijcai2027_canonical_tables.md](file:///d:/research/cscwd_tri_voi/results/ijcai2027_canonical_tables.md) | [results/ijcai2027_canonical_tables.tex](file:///d:/research/cscwd_tri_voi/results/ijcai2027_canonical_tables.tex)
- 4 张核心论文配图（[generate_ijcai_figures.py](file:///d:/research/cscwd_tri_voi/generate_ijcai_figures.py) & [verify_theory_package.py](file:///d:/research/cscwd_tri_voi/verify_theory_package.py)）：
  - `Fig. 1`: [fig1_mdp_mechanism_and_regret.pdf](file:///d:/research/cscwd_tri_voi/results/figs/fig1_mdp_mechanism_and_regret.pdf)
  - `Fig. 2`: [fig2_theory_verification.pdf](file:///d:/research/cscwd_tri_voi/results/figs/fig2_theory_verification.pdf)
  - `Fig. 3`: [fig3_efficiency_and_c_ask_phase.pdf](file:///d:/research/cscwd_tri_voi/results/figs/fig3_efficiency_and_c_ask_phase.pdf)
  - `Fig. 4`: [fig4_mediq_disjoint_clinical_blind.pdf](file:///d:/research/cscwd_tri_voi/results/figs/fig4_mediq_disjoint_clinical_blind.pdf)
- 理论证明与数值伴随验证：[docs/theory/ijcai2027_three_theorem_proofs.md](file:///d:/research/cscwd_tri_voi/docs/theory/ijcai2027_three_theorem_proofs.md) | [verify_theory_package.py](file:///d:/research/cscwd_tri_voi/verify_theory_package.py)
- 离线归因、Oracle 审计与去重切片：[results/attribution_and_dedup_audit.md](file:///d:/research/cscwd_tri_voi/results/attribution_and_dedup_audit.md)
- 预注册 Git Tags 与只读归档分支：
  - Tags：`prereg-blind-v3`、`prereg-mediq-test-disjoint`（[results/mediq_test_disjoint_eval.md](file:///d:/research/cscwd_tri_voi/results/mediq_test_disjoint_eval.md)）
  - 只读归档分支：`archive/v1-tri-voi-l2d` (`76bd911`)、`archive/v2-mdp-oracle-diag` (`8fcfbf8`)、`archive/full-history-snapshot` (`9b7ea8c`)
