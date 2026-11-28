# Tri-VoI-MDP：面向三方协作决策的两阶段信息价值控制与 Bellman 拥塞定价框架

[English](./README.md) | **简体中文**

**统一主方法（IJCAI 2027）**：**`Tri-VoI-MDP`（`Deploy-Self`，100% 零 Oracle）**——面向 **用户（User）**、快速大语言模型**全科助手（Generalist）**与受队列容量约束的**领域专家（Specialist，`LLM Specialist` 或仿真接管模型）**的可部署两阶段序贯控制与排队决策框架。

相比于此前预注册的 5-Call 静态前代方法（`Tri-VoI-L2D`，在仓库中完整保留以作谱系对比），`Tri-VoI-MDP` 实现了三项核心升级：
1. 将 5-Call 自洽性探测精简为 **2-Call 粗充分统计量（coarse sufficient statistic）**，在 `Benchmark v1-Clean` 与 `Benchmark v2` 上将实测 LLM Token 开销降低 **53.7%–57.8%**。
2. 将依赖模拟器真实反馈 `oracle_revealed` 的澄清后信号替换为严格本地可部署的 **`post_missing` 自监督信号**（解析失败时仅回退至 `obs_missing`，全程零访问模拟器真值）。
3. 将近视的静态 Naor 队列阈值替换为服务器拥塞条件下的**有限时域周期排空 Bellman 动态规划影子价格**（$\Delta V(q)$，见 **Theorem 4 & Proposition 5**）。

---

## 方法谱系与理论保证

完整形式化定理、证明及机器核验数值伴随结果见 [ijcai2027_three_theorem_proofs.md](./docs/theory/ijcai2027_three_theorem_proofs.md)、[GLOSSARY.md](./GLOSSARY.md) 与 [PAPER_STORY_AND_STATUS.md](./PAPER_STORY_AND_STATUS.md)。

| 方法 / 组件 | 论文角色定位 | 适用范围与理论保证 |
|:---|:---|:---|
| **`Tri-VoI-L2D`**（`Theorems 1–3`） | 前代 5-Call + Naor 静态基线 | 确立两阶段 $\mathcal{O}(\log N)$ 单调澄清/转交策略与次模信息价值界；在 `v1-Clean / Qwen3.5` 上冻结复现（`[5.220, 5.142, 3.792]`）。 |
| **`Tri-VoI-MDP`**（`Theorem 4 & Proposition 5`） | **本文唯一主方法（`Deploy-Self`，零 Oracle）** | 当 $\gamma > 0$ 时，证明有限时域 Bellman DP 在队列长度 $q$ 上的最优放弃单调阈值结构（`Theorem 4`），并给出基于 2-Call 粗划分与多拥塞/代价场景的最坏情况极小化极大后悔界（`Proposition 5`）。 |
| **`MDP (Oracle-NoInfo Diag)`** | **仅作消融诊断对比** | 旧版在 `ASK` 后读取 `not oracle_revealed` 的分支；仅隔离展示于显式诊断列（`Sections 9A, 9B, 10`），不计入任何主结论。 |

---

## 核心实证结论（100% 零 Oracle 单一事实源）

以下全部关键指标均自动汇总生成于 [ijcai2027_canonical_tables.md](./results/ijcai2027_canonical_tables.md) 与 [ijcai2027_canonical_tables.tex](./results/ijcai2027_canonical_tables.tex)：

1. **高拥塞下跨 13 套件的普适 `+MDP` 机制增益（`Section 0A`，`Cap=1/3, c=1.5`）**：
   将静态 Naor 阈值替换为动态 Bellman DP 影子价格后，`CC+Naor -> CC+MDP`（均值 `+0.355`）与 `ACL+FD+Naor -> ACL+FD+MDP`（均值 `+0.443`）在 **13/13** 个评估套件上全部取得正增益；`Tri-VoI-L2D -> Tri-VoI-MDP`（均值 `+0.429`）与 `JG+Naor -> JG+MDP`（均值 `+0.429`）在 **12/13** 个套件上取得正增益。
2. **跨场景最低极小化极大后悔（`Proposition 5`，`Section 0B`）**：
   - 在 **26 个严格配对 `2C-Self` 单元**（`13 套件 × {Cap=2, Cap=1/3}`、`c=1.5`、零 Oracle、相同 DP 定价）中，`Tri-VoI-MDP` 取得最高平均效用（**3.617**）、最多胜场（**10/26**）、最低平均后悔（**0.099**）与最低最坏情况后悔（**0.464**；对比 `CC+MDP` `0.625`、`ACL+FD+MDP` `0.764`、`JG+MDP` `0.820`）。
   - 在 **40 个 `c_ask` 澄清代价扫描单元**（`c_ask ∈ {0.5, 1.5, 3.0, 4.5}`）中，`Tri-VoI-MDP` 取得最高平均效用（**3.380**）与最低最坏情况后悔（**0.526**；对比 `ACL+FD+MDP` `0.830`、`JG+MDP` `0.927`、`CC+MDP` `1.330`）。
   - 在 **56 格 2D 灵敏度包络（`Table 3`）**中，`Tri-VoI-MDP` 在 **5/7** 个套件上排名 `#1`（包括 `v1-Clean Qwen3.5` **4.623** / 最坏后悔 **0.404**、`v2 DeepSeek-V3.2` **4.494** / **0.195** 以及全部三个 `Public-Tri` 套件）。
3. **Token 效率提升（`Section 6`）**：
   相比 5-Call 完整探测 `Tri-VoI`，实测 SQLite Token 消耗下降 **53.7%–57.8%**（`v1-Clean Qwen3.5`：`4952 -> 2111`，`-57.4%`；`v1-Clean DeepSeek-V3.2`：`4441 -> 2056`，`-53.7%`；`v2 Qwen3.5`：`4967 -> 2097`，`-57.8%`），同时维持或提升净效用。
4. **预注册跨器官系统盲测临床评估（`MediQ-Test-Disjoint`，`Section 10`）**：
   在 Git 标签 `prereg-mediq-test-disjoint` 预注册（`N_cal=60`，`N_test=180`，`PY_DIFF_LEN=0`，`0` 解析失败回退，在线 API 花费 `¥3.55`）。在跨组真实 LLM Specialist 委派（`Qwen3.5-35B-A3B -> DeepSeek-V3.2`）下，`Tri-VoI-MDP` 在 `Cap=2, c=1.5` 达到 **4.266**（原始医学准确率 `81.7%` / 越界转交召回率 `85.1%`，`1823` Token），在 `Cap=1/3, c=1.5` 达到 **3.588**（领先 `Tri-VoI-L2D` `+0.679`，`p=0.0137`；`c=3.0` 时领先 `+0.785`，`p=0.0057`），且高拥塞下所有四组 `+MDP` 升级均显著优于静态 Naor（`p < 0.05`）。
5. **诚实边界披露（`Sections 8, 9D, 9E, 10B`）**：
   - 在严格配对的 `2C-Self` 同预算同 DP 定价对比中（`Section 9E`），`Tri-VoI-MDP` 在单个单元上并**不**对强基线 `Base+MDP (2C-S)` 构成统计学显著碾压（`p=0.2036–0.9810`）；在 `v2 (Qwen3.5)`、`v2 (Hy4-preview)` 与 `MediQ-Test-Disjoint Setup B` 部分单元上，`CC+MDP (2C-S)`、`ACL+FD+MDP (2C-S)` 或 `JG+Naor` 领先具体单格。
   - `Benchmark v3`（`Section 8`，`prereg-blind-v3`）存在已披露的属性模板重合（每个未见领域包含 8 个模板：`50.0%` 归一化文本重合、`35.4%` 完整三元组重合）；在严格零重合切片（`Section 9D` 中 `72` 与 `40` 样本）上与配对 `2C-S` 基线持平（`p=0.1546–1.0000`）。

---

## 仓库目录结构

```text
.
├── controller.py                        # 核心 Tri-VoI-MDP (Deploy-Self)、Tri-VoI-L2D 与基线策略控制器
├── external_baselines.py                # 配对 2C-Self 与 5C 外部基线 (KL-PBD+, ACL-2026-VoI, JG, CC)
├── probe_profiles.py                    # 离线/在线 LLM 广义探测与 post_missing 自监督信号提取器
├── llm_client.py                        # SiliconFlow OpenAI 兼容客户端、费用统计与 SQLite 缓存
├── benchmark.py / public_tri.py         # Benchmark v1-Clean/v1-Noisy、v2 与 Public-Tri (MediQ 执业边界分诊)
├── benchmark_v3.py                      # 未见模板领域 (Benchmark v3) 与 MediQ-Holdout 划分
├── mediq_test_disjoint.py               # 跨器官系统、长度匹配的 MediQ-Test-Disjoint 数据构建器
├── generate_ijcai_tables.py             # 自动生成已校验的 results/ijcai2027_canonical_tables.{md,tex}
├── verify_theory_package.py             # 零 API 调用验证 Theorems 1–4 与 Proposition 5
├── run_deploy_self_canonical_upgrade.py # Sections 0–8 的离线零 Oracle 升级重算脚本
├── run_attribution_and_dedup_audit.py   # Section 9 离线归因（配对 2C-S 对比、post_missing 审计、v3 切片）
├── run_mediq_test_disjoint_eval.py      # Section 10 主运行脚本（MediQ-Test-Disjoint）
├── docs/theory/                         # 理论证明文档与数值验证 JSON/MD 文件
└── results/
    ├── README.md                        # 活跃结果文档与归档文件说明指南
    ├── ijcai2027_canonical_tables.md    # 主 Markdown 结果总表（Sections 0–10）
    ├── ijcai2027_canonical_tables.tex   # 对应 Sections 1–8 的 LaTeX 表格
    ├── attribution_and_dedup_audit.md   # Section 9 详细离线归因与信号审计报告
    ├── mediq_test_disjoint_eval.md      # 预注册 MediQ-Test-Disjoint 临床评估报告
    ├── deploy_self_canonical_upgrade.json
    └── archive/                         # 已替代的早期 Pre-MDP 与 Oracle 诊断阶段性历史报告
```

---

## 离线复现与主表重生成（`live=0`, `¥0.00`）

所有代码、表格与理论一致性校验均可通过仓库内缓存的 SQLite 响应与已验证 JSON 输出完全离线运行：

```bash
python verify_theory_package.py
python run_deploy_self_canonical_upgrade.py
python run_attribution_and_dedup_audit.py --offline-only
python generate_ijcai_tables.py
```

若需配置环境或针对 SiliconFlow 在线复跑 `MediQ-Test-Disjoint` 盲测，请参阅 [mediq_test_disjoint_eval.md](./results/mediq_test_disjoint_eval.md)。
