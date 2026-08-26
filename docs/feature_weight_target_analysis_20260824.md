# 2026 量化交易研究大赛：Feature、Weight 与 Target 分析报告

日期：2026-08-24
依据：赛题 PDF v1.3、官方数据说明、13,227,692 行训练数据全量画像

## 1. 执行结论

当前模型确实还缺少一层重要的“问题定义”：323 个匿名特征此前主要被当作普通表格列使用，没有系统区分它们可能描述的是价格、成交、波动、流动性、路径还是跨标的状态，也没有充分利用 `weight`、`target` 幅度和 47 个 responder 所表达的不同监督信息。

本次复核后的核心结论是：

1. `target` 不是普通收益率，而是未来固定窗口内的风险调整表现类目标。正负号是方向，绝对值是表现强度。接近 0 通常表示未来方向性表现弱；绝对值较大表示较强的正向或反向表现。
2. `weight` 不是信号，也不表示涨跌方向。它是评分时的样本重要性，官方说明主要综合未来窗口的真实成交活跃度、交易摩擦和相对评估贡献，一般真实成交额越高、摩擦越低，权重越高。
3. `feature_*` 的逐列金融名称、窗口和公式没有公开，不能声称 `feature_240` 就是成交量、`feature_286` 就是动量。我们能建立的是“统计行为角色”，不是恢复真实字段名。
4. `weight` 与 `target` 的相关系数只有 `-0.0061`，说明官方没有简单按 target 正负或大小赋权。
5. `weight` 首先具有很强的标的固定差异。只使用早期分区学到的 `asset_id` 平均权重，在后两个时间分区上可以解释 `77.29%` 的权重方差；加入 10 个可见匿名特征代理后，探索性诊断模型可以解释约 `87.50%`。这说明当前特征里存在活跃度、流动性或摩擦状态代理，但不等于恢复了官方权重公式，也不是正式 target 模型成绩。
6. 单个 feature 与 `target` 的线性关系都很弱，最大绝对 Pearson 相关只有约 `0.036`；与 `abs(target)` 的最大相关只有约 `0.010`。真正的信号更可能来自非线性、特征组合、横截面相对位置、滞后变化和市场状态条件关系。
7. `abs(target)` 最大的 5% 样本贡献评分分母 `sum(weight * target^2)` 的 `17.36%`，最大的 10% 贡献 `31.31%`。幅度建模不能缺失，但也不能直接使用 `target^2` 重新加权训练，因为官方误差项的样本权重仍然只是 `weight`。
8. 47 个 responder 明显包含至少两类未来响应轴：一类与 target 方向高度相关，另一类与 weight/流动性状态高度相关。它们适合做训练阶段辅助监督，不能在测试时直接作为输入。
9. 已按“`feature -> predicted weight/responders -> target`”实现第一版严格前向二阶段模型。历史 3 折均提升；对当前最强模型的冻结后半段增量为 `+0.000190`，全段本地分数从 `0.00612476` 提升到 `0.00641096`。测试候选已生成并通过结构审计，但没有线上分数，因此仍标记为候选而不是已确认的新最优。

## 2. 证据分级

为避免对匿名数据过度解释，本文使用以下标记：

- **官方明确**：PDF 或官方数据说明直接给出的定义。
- **数据验证**：从当前发布的 9 个训练分区全量或严格时间前向诊断中得到。
- **合理推断**：与官方描述及统计结果一致，但没有匿名字段映射，不能当作真实金融定义。
- **尚未确认**：主办方没有公开，当前数据也不能唯一确定。

## 3. 官方字段到底表示什么

### 3.1 索引字段

| 字段 | 官方含义 | 建模作用 |
|---|---|---|
| `row_id` | 样本唯一标识 | 只用于行对齐和提交，不应作为金融信号 |
| `time_id` | 匿名时间顺序，值越大越靠后 | 用于严格前向切分、因果滚动状态和顺序推理；不对应真实日历 |
| `asset_id` | 匿名标的编号 | 表示同一标的的连续样本，可用于标的固定效应和分资产状态；不包含真实名称或规模含义 |

### 3.2 `feature_*`

**官方明确**：323 个匿名数值特征全部由当前及历史可见信息构造，不包含未来目标信息。可能覆盖以下行为类别，但官方不公开每一列的真实名称、公式和计算窗口。

| 可能的行为类别 | 大概描述什么 | 对 target 可能有什么作用 | 对 weight 可能有什么作用 |
|---|---|---|---|
| 价格变化 | 局部上涨、下跌、趋势和反转状态 | 判断未来方向和趋势持续/反转 | 通常不是直接权重来源 |
| 成交状态 | 成交强弱、活跃程度及其变化 | 活跃时价格信号的有效性可能不同 | 很可能是未来成交活跃度的当前代理 |
| 波动结构 | 当前及历史不确定性、波动扩张或收缩 | 决定风险调整后的幅度和状态切换 | 可能影响摩擦和可交易性 |
| 流动性状态 | 市场深度、冲击成本、买卖难易程度的匿名代理 | 决定信号能否兑现、是否需要收缩 | 与“成交额更高、摩擦更低则权重更高”的官方描述最相关 |
| 路径形态 | 过去窗口内趋势是否平滑、是否有回撤或跳变 | 区分同样终点变化背后的不同路径风险 | 可能反映交易摩擦或风险状态 |
| 跨标的关系 | 市场共同因子、相对强弱、联动或分歧 | 判断共同方向、个体偏离和横截面机会 | 可能反映整体市场活跃状态 |

**尚未确认**：无法把上述类别逐一映射到具体 `feature_XXX`。因此本项目后续应把每个 feature 标记为“统计角色”，例如：方向候选、幅度候选、weight/流动性候选、稳定状态变量、短时变化变量或弱单变量候选，而不是伪造金融字段名称。

当前 323 列的逐列画像见：

- [`feature_profile.csv`](../results/feature_weight_target_analysis_20260824/feature_profile.csv)
- [`top_target_related_features.csv`](../results/feature_weight_target_analysis_20260824/top_target_related_features.csv)
- [`top_weight_related_features.csv`](../results/feature_weight_target_analysis_20260824/top_weight_related_features.csv)

逐列画像已经包含缺失率、均值、标准差、与 target/abs(target)/weight 的相关性，以及 9 个时间分区中的均值、波动和符号稳定性。它是目前可诚实建立的第一版 feature role dictionary。

### 3.3 `responder_*`

**官方明确**：47 个 responder 是训练阶段可见、由未来不可见区间构造的辅助响应变量，可能覆盖不同预测窗口和收益、风险、路径、流动性/摩擦等响应维度。测试阶段不提供。

因此：

- 可以用作多任务学习的辅助标签；
- 可以训练 `feature -> responder` 的模型，再用严格时间前向 OOF 预测形成潜在状态；
- 不能把训练数据中的真实 responder 直接放进线上模型；
- 不能在当前样本预测时访问未来 responder。

### 3.4 `target`

**官方明确**：`target` 是当前状态之后某一固定预测窗口内的风险调整表现类目标。

- `target > 0`：未来风险调整表现偏正向；
- `target < 0`：未来风险调整表现偏反向；
- `abs(target)` 小：未来方向性表现弱，通常更接近“没有可用强信号”；
- `abs(target)` 大：未来存在更强的正向或反向表现。

这里需要两点修正：

1. 小 target 不是“这行完全没用”。如果模型在小 target、高 weight 样本上输出很大的错误预测，同样会被评分惩罚。合理策略通常是低置信度时向 0 收缩。
2. 大 target 在预测时不可见。模型要学习的是条件期望 `E[target | 当前及历史信息]`，不能事后挑选真实大 target 样本。

### 3.5 `weight`

**官方明确**：`weight` 是评分权重，主要综合未来窗口的真实成交活跃度、交易摩擦和样本对整体评估的相对贡献。一般真实成交额越高、摩擦越低，权重越高。

它的含义不是：

- 不是 target 的一部分；
- 不是涨跌方向；
- 不是模型应该输出的置信度；
- 不是测试时可见特征。

它的正确用法是：

- 训练 target 模型时作为样本权重；
- 验证时严格按官方 Weighted Zero-Mean R2 计算；
- 可以作为训练阶段辅助标签，学习可见的流动性/活跃度状态代理；
- 测试时只能使用模型从 `feature_*` 和历史状态预测出的代理，不能访问真实 weight。

## 4. 评分公式如何改变建模重点

官方指标为：

```text
Score = 1 - sum(weight * (target - prediction)^2)
            / sum(weight * target^2)
```

全零预测的分数恰好是 0。因此模型只有在加权平方误差小于全零基线时才获得正分。

这带来四个直接结论：

1. **方向错误很贵**：真实 target 为正而预测为负，或反之，会明显增加平方误差。
2. **幅度必须校准**：方向对但幅度过大，也可能比小幅、稳健的预测更差。当前强模型的预测标准差约 `0.071`，远小于 target 标准差约 `1.091`，本质上是在弱信号环境下做强收缩。
3. **weight 决定误差的重要性**：同样大小的误差，在高 weight 样本上惩罚更大。
4. **大 target 提供更大的潜在可解释空间**：对全零基线而言，每行分母贡献是 `weight * target^2`。但正式训练的平方误差权重应仍为 `weight`，不能额外乘 `target^2`，否则会改变比赛真正优化的目标并放大尾部过拟合。

## 5. Target 的实际分布

**数据验证，全量 13,227,692 行：**

| 统计量 | 数值 |
|---|---:|
| 均值 | 0.00888 |
| 中位数 | 0.00638 |
| 标准差 | 1.09104 |
| 平均绝对值 | 0.93121 |
| 最小值 | -2.23563 |
| 最大值 | 2.23577 |
| 5% / 95% 分位数 | -1.72722 / 1.74703 |
| 1% / 99% 分位数 | -2.02299 / 2.03381 |

分布整体接近零中心且正负较对称，但不是简单的高斯小噪声。`abs(target)` 最大的：

| 样本范围 | `sum(weight * target^2)` 占比 |
|---|---:|
| 最大 5% | 17.36% |
| 最大 10% | 31.31% |

这说明模型需要同时解决三个问题：

- 是否存在可预测信号；
- 信号方向是什么；
- 预测应该有多大，还是应该收缩到 0。

## 6. Weight 到底由什么决定

### 6.1 官方层面的答案

官方只公开了生成原则，没有公开精确公式：

```text
未来窗口真实成交活跃度 + 交易摩擦 + 样本相对评估贡献
```

一般而言：

```text
真实成交额更高、交易摩擦更低 -> weight 更高
```

因此任何“weight 等于某几个 feature 的固定公式”的说法都没有依据。

### 6.2 数据层面的答案

全量数据中：

| 统计量 | 数值 |
|---|---:|
| weight 均值 | 2.31635 |
| weight 标准差 | 1.71069 |
| corr(weight, target) | -0.00609 |

不同 asset 的平均 weight 差异非常大：

| asset | 平均 weight | 评分分母贡献占比 |
|---:|---:|---:|
| 5 | 5.892 | 16.94% |
| 8 | 5.461 | 15.37% |
| 11 | 3.794 | 10.28% |
| 6 | 2.709 | 7.76% |
| 14 | 2.545 | 7.26% |
| 7 | 0.841 | 2.46% |
| 13 | 0.993 | 2.90% |

按时间顺序使用 partition 0-6 拟合、partition 7-8 验证的探索性诊断结果如下：

| weight 代理模型 | 验证 R2 |
|---|---:|
| 仅使用历史 asset 平均 weight | 0.7729 |
| asset + 前 5 个 weight 候选 feature | 0.8427 |
| asset + 前 10 个候选 | 0.8750 |
| asset + 前 20 个候选 | 0.8781 |
| asset + 前 50 个候选 | 0.8937 |

这里的 LightGBM 只用于结构诊断，训练和验证分别取确定性哈希约 1/20 样本；asset-only 指标使用全量验证行精确计算。候选 feature 排名来自全量探索画像，而不是嵌套在训练分区内重新筛选，因此这些 R2 用来说明“存在较强代理结构”，不能当作完全无偏的未来泛化成绩。正式 predicted-weight 实验仍需在每个前向折内部重新筛选。结果文件：

- [`weight_structure_summary.json`](../results/feature_weight_target_analysis_20260824/weight_structure_summary.json)
- [`weight_feature_structure.csv`](../results/feature_weight_target_analysis_20260824/weight_feature_structure.csv)
- [`weight_proxy_validation.csv`](../results/feature_weight_target_analysis_20260824/weight_proxy_validation.csv)

### 6.3 与 weight 关系最强的可见 feature

| feature | 全局 corr(weight) | asset 内去均值 corr(weight) | 稳定性 |
|---|---:|---:|---|
| `feature_240` | -0.620 | -0.265 | 9 个分区均为负 |
| `feature_207` | +0.504 | +0.407 | 9 个分区均为正 |
| `feature_100` | -0.438 | -0.520 | 9 个分区均为负 |
| `feature_175` | -0.435 | -0.457 | 9 个分区均为负 |
| `feature_041` | +0.423 | +0.312 | 9 个分区均为正 |
| `feature_101` | -0.413 | -0.554 | 9 个分区均为负 |
| `feature_104` | -0.368 | +0.052 | 全局关系主要来自 asset 间差异 |
| `feature_016` | +0.345 | +0.198 | 9 个分区均为正 |
| `feature_040` | +0.335 | +0.181 | 9 个分区均为正 |
| `feature_018` | +0.271 | +0.262 | 9 个分区均为正 |

**合理推断**：这些列是活跃度、流动性、摩擦或 asset 结构的候选代理。`feature_100`、`feature_101`、`feature_175`、`feature_207` 在去除 asset 固定差异后仍然有较强关系，更值得用于流动性状态建模。`feature_104` 的全局相关很高，但 asset 内关系很弱，主要像是一个标的层面的结构代理。

**不能推断**：不能根据正负相关把它们直接命名为成交量、点差、换手率或冲击成本。匿名映射仍然未知。

## 7. Feature 与 Target 的信号结构

与 target 线性相关性最高的 feature 为：

| feature | corr(target) | 9 个分区方向 |
|---|---:|---|
| `feature_286` | -0.03595 | 全部为负 |
| `feature_284` | -0.03525 | 全部为负 |
| `feature_148` | -0.03218 | 全部为负 |
| `feature_157` | -0.03195 | 全部为负 |
| `feature_039` | -0.02953 | 全部为负 |

这几列虽然弱，但时间方向较稳定，适合保留为方向基线。更关键的是：

- 323 个 feature 中，没有任何单列表现出强 target 线性关系；
- 与 `abs(target)` 的最大绝对相关只有约 `0.0100`；
- 295 个 feature 在第一版画像中都属于“弱单变量或待挖掘非线性”；
- 28 个 feature 更明显地属于 weight/状态代理候选。

**合理推断**：当前任务不是“找到一个神奇指标”即可解决。主要增量更可能来自：

- 同一 asset 内 feature 的滞后、变化量和滚动标准化；
- 同一 time_id 下 15 个 asset 的 rank、去均值和相对强弱；
- 多个 feature 的非线性交互；
- 波动、流动性和市场一致性状态下的条件模型；
- 对弱置信度预测做动态收缩。

## 8. Responder 提供了什么额外信息

与 target 相关最高的 responder：

| responder | corr(target) | 建议角色 |
|---|---:|---|
| `responder_03` | +0.810 | target 方向/共同未来表现辅助轴 |
| `responder_28` | +0.686 | 另一窗口或响应维度的方向辅助轴 |
| `responder_02` | +0.557 | 方向辅助轴 |
| `responder_29` | +0.546 | 方向辅助轴 |
| `responder_18` | +0.431 | 方向与状态混合辅助轴 |

与 weight 相关最高的 responder：

| responder | corr(weight) | 建议角色 |
|---|---:|---|
| `responder_37` | -0.616 | 未来流动性/摩擦状态辅助轴候选 |
| `responder_36` | -0.614 | 同上 |
| `responder_35` | -0.613 | 同上 |
| `responder_34` | -0.611 | 同上 |
| `responder_33` | -0.608 | 同上 |
| `responder_32` | -0.605 | 同上 |
| `responder_31` | -0.596 | 同上 |

这些名称仍是统计角色，不是真实金融命名。最合理的使用方式是共享编码器、多任务头，或者严格 OOF 的 responder 预测堆叠；真实 responder 永远不能作为测试输入。

## 9. 二阶段辅助状态模型：已实现结果

本轮把核心路线明确改成：

```text
323 features + asset_id
        |
        +--> predicted weight
        +--> predicted responder_00..46
                       |
                       +--> two-stage target prediction
                                      |
current regime + XGBoost model -------+--> residual correction
```

### 9.1 泄漏边界

这条路线只有在以下边界成立时才有效：

- 辅助模型只在更早时间段用真实 weight/responders 训练；
- target 模型看到的是辅助模型预测值，不是同一行真实 weight/responders；
- 真实 weight 只作为 target 训练损失和验证指标的样本权重；
- test 只读取 `row_id/time_id/asset_id + feature_000..322`；
- 官方 test 只用于推理，不参与选型、gamma 拟合或分桶调参。

主折时间协议为：

```text
aux train       688480..776479
aux valid       777480..787479
target train    788480..846479
calibration     847480..867479
outer validate  868480..888479
各段 purge gap  1000
```

### 9.2 当前模型结构

第一阶段同时训练两个辅助模型：

- 多输出 Ridge：输入全部 323 个 feature 和 asset one-hot，同时预测 `weight + 47 responders`；
- 共享 GPU MLP：两层共享表示，同样输出 48 个预测辅助状态。

第二阶段使用 Ridge 主干和 LightGBM residual，并完成以下消融：

- raw feature only；
- predicted auxiliary only；
- predicted weight；
- predicted liquidity responders；
- predicted direction responders；
- 全部 predicted responders；
- raw 与上述状态的组合。

calibration 选定结构和 residual weight/shrink 后，将 target train 与 calibration 合并重训；所有选择保持冻结，再预测 outer。

### 9.3 辅助变量到底能不能预测

主折 outer 的代表性结果为：

| 辅助变量 | feature 预测相关性 | 结论 |
|---|---:|---|
| `weight`，Ridge | `0.9589` | 可较强重建流动性/重要性状态 |
| `responder_31..37`，Ridge | 约 `0.986..0.990` | 这组高度可预测，主要像稳定的流动性/摩擦响应轴 |
| `responder_18`，Ridge | 约 `0.457` | 存在中等强度可预测方向/状态信息 |
| `responder_02/03/28/29` | 约 `0.01..0.03` | 虽与真实 target 高相关，但从当前 feature 很难单独预测 |

这里最重要的认识是：`corr(true responder, target)` 高，不等于 `predicted responder` 一定能提高 target。建模价值取决于 responder 能否被当前可见 feature 稳定预测，以及其预测误差是否仍包含主模型没有利用的残差信息。

### 9.4 前向验证结果

每个历史折都只能根据自己的 calibration 选择辅助结构。重训后，calibration 预选模型相对 refit raw control 的结果为：

| 折 | calibration 预选结构 | 直接替代 full 增量 | untouched 后半增量 |
|---|---|---:|---:|
| fold0 | `predicted_aux_only` | `+0.000277` | `+0.000333` |
| fold1 | `predicted_aux_only` | `+0.001122` | `+0.001560` |
| fold2 | `raw_plus_all_aux` | `+0.000399` | `+0.000440` |

历史 3/3 折 full 增量为正，3/3 折 untouched 后半也为正。用每折前半拟合辅助 residual 系数并冻结到后半时，同样为 3/3 折提升。

对当前 `regime composite + XGBoost residual` 强基线，最终采用：

```text
auxiliary_delta = two_stage_prediction - raw_auxiliary_control
candidate = current_best - 0.4363287926 * auxiliary_delta
```

| 区间 | 当前基线 | 新候选 | 增量 |
|---|---:|---:|---:|
| gamma 拟合前半 `868480..878479` | `0.00623861` | `0.00662274` | `+0.00038414` |
| gamma 冻结后半 `878480..888479` | `0.00601272` | `0.00620254` | `+0.00018982` |
| outer 全段 | `0.00612476` | `0.00641096` | `+0.00028620` |

前半拟合 gamma 为 `-0.4363`，后半独立最优 gamma 为 `-0.3383`，方向一致。负号不表示 responder 与 target 的金融含义反转；它表示相对于已经很强的 current-best 基线，`two_stage - raw control` 在该时间段是一个反向残差修正方向。历史弱 raw control 上的辅助系数仍为正，因此不能把这个负号固化解释成 responder 的普遍方向。

### 9.5 稳定性与当前状态

- 细时间块：5/8 提升；
- asset：11/15 提升；
- `abs(target)` 五分位：5/5 提升；
- weight 五分位：最高两个 weight 桶提升，低 weight 三个桶下降；
- 测试集 3,217,458 行全部完成推理，无 NaN/Inf；
- submission 行序、唯一性、ZIP 内容和公式复算均通过独立结构审计。

因此当前结论是：这条二阶段路线已经产生了可信的本地增量和可提交候选，但细时间块仍有 3/8 下降，且没有线上分数。候选不覆盖原推荐版本，优先用于一次独立线上对照。

## 10. 模型改进计划

### Phase 0：二阶段 weight/responder stacking（第一版已完成）

核心目标不是分别做一个 weight 小实验和 responder 小实验，而是把完整链路作为主干：

```text
feature
  -> predicted weight + predicted responders
  -> target model / residual model
  -> current-best residual fusion
```

第一版已经完成 Ridge + 共享 MLP 辅助预测、target 消融、calibration 后重训、3 个历史折、最新冻结 holdout 和测试候选。下一版不再重复验证“这条路线能否工作”，而是解决它当前的主要不足：方向 responder 可预测性弱、低 weight 桶下降、时间块仍有漂移。

### Phase 1：按可预测性重构 responder 表征

当前 48 个辅助任务等权使用，会让网络把容量浪费在几乎不可预测的 responder 上。下一轮：

- 按严格 aux-valid 相关性把 responder 分成高、中、低可预测三组；
- 对高度共线的 `31..37` 做 PCA/低秩流动性状态，而不是重复预测七条近似轴；
- 对 `02/03/28/29` 不追求逐列拟合，改为学习它们与 target 相关的低维投影；
- 对每个辅助头使用独立 loss weight、dropout 和 shrink；
- 只允许依据 aux-valid 选择任务权重，不能依据 outer target 回调。

验收：相比第一版 predicted auxiliary，历史多数折和最新 frozen holdout 均提升，且 gamma 符号更稳定。

### Phase 2：Predicted-weight 门控和高重要性优化

当前辅助 residual 在最高两个 weight 桶提升，但低 weight 三个桶下降。下一轮直接利用 feature 预测的 weight 状态：

- 比较连续 predicted-weight gate、分位数 gate 和动态 shrink；
- gate 必须来自 predicted weight，验证/test 不得读取真实 weight；
- 将流动性 responder 的低秩状态与 predicted weight 联合建模；
- 正式 target loss 仍使用训练行真实 weight，不额外乘 target 幅度；
- 检查提升是否来自高 weight 样本，同时限制低 weight 桶的负迁移。

验收：高 weight 桶保持正增量，低 weight 桶总体不明显恶化，full score 不依赖单一 asset。

### Phase 3：建立因果 Feature Role Dictionary

目标：把“323 个匿名列”转成有统计概念的输入组。

每个 feature 增加以下画像：

- asset 内自相关和半衰期；
- 同 time_id 横截面方差与 rank 稳定性；
- asset 间方差占比和 asset 内方差占比；
- lag、delta、滚动 mean/std/min/max/range；
- 缺失率、连续缺失长度和缺失状态切换；
- 与 target、sign(target)、abs(target)、weight 和 responder 组的 OOF 关系；
- 9 个分区以及更细时间块中的方向稳定性；
- 特征间相关聚类，减少重复列。

只使用两类时序安全变换：

1. 每个 asset 内仅基于当前及过去的 causal rolling/expanding 变换；
2. 当前 time_id 已同时可见的 15 个 asset 之间做横截面 rank、z-score、demean 和分歧度。

第一轮模型只比较：raw、raw+横截面、raw+causal lag/rolling、三者合并，确认每类特征的独立增量。

### Phase 4：Target 方向、强度与弱信号联合建模

目标：不再把 target 只当作一个无结构连续值。

建立三个头：

- 方向头：预测 `P(target > 0 | x)`；
- 幅度头：预测正向和负向条件下的 `E[abs(target) | x]`；
- 弱信号/置信度头：预测 target 是否落入接近 0 的 OOF 阈值区间。

软组合公式：

```text
E[target | x]
= P(target > 0 | x) * E[abs(target) | target > 0, x]
- P(target < 0 | x) * E[abs(target) | target < 0, x]
```

最终输出再用 OOF 数据做斜率校准和动态 shrink。不要使用硬分类符号，也不要把真实 `abs(target)` 当作线上置信度。

实验顺序：

1. 直接加权回归基线；
2. 方向头单独验证；
3. 正负条件幅度头；
4. 弱信号门控；
5. 与直接回归做 OOF 残差融合。

### Phase 5：状态专家和序列模型

目标：让模型在不同市场状态下采用不同的信号与收缩程度。

状态至少包括：

- 高/低波动；
- 高/低流动性或活跃度代理；
- 横截面一致上涨、一致下跌、分歧；
- 趋势、反转和跳变路径；
- 高/低预测置信度。

先使用轻量 LightGBM/CatBoost 专家和连续门控，再评估 causal TCN。训练可以使用 WSL GPU，但最终私榜只有 4 核 CPU、12 GB 内存且无 GPU，因此最终序列模型必须压缩，并通过官方 Time-Series API 的 CPU 顺序推理测试。

### Phase 6：最终融合与替换标准

只允许使用严格 OOF 预测拟合融合权重。最终候选至少包含：

- 当前 regime composite；
- 当前 XGBoost residual；
- 新的方向/幅度模型；
- weight-state residual；
- responder multi-task residual；
- causal sequence residual。

替换当前推荐模型的最低标准：

1. 多数前向折提升；
2. 独立冻结 holdout 提升；
3. 不是只靠一两个高 weight asset；
4. `abs(target)` 各桶没有明显异常过拟合；
5. 8 个或更多细时间块的提升比例稳定高于当前候选的 5/8；
6. CPU Time-Series API 完整运行、无未来信息、无超时、无 NaN/Inf。

## 11. 推荐的近期实验顺序

Predicted-weight 门控、纯可预测性筛选、低秩 responder 压缩和 target 残差相关筛选已经完成严格验证，均未超过当前候选。后续不继续调这些门槛，按收益、风险和工程成本改为：

| 优先级 | 实验 | 原因 |
|---:|---|---|
| P0 | 横截面 rank/demean + asset 内 causal delta/rolling | 三种辅助筛选都失败，说明当前瓶颈更可能在输入表征，不在 48 维输出的简单删减 |
| P0 | 方向头 + 正负条件幅度头 + 弱信号头 | 直接利用 target 的符号、绝对值和近零区间结构，避免只做一个粗糙连续回归 |
| P0 | 增加历史前向折和 purge-gap 敏感性 | 当前只有 3 个历史折，仍不足以确认跨状态稳健性 |
| P1 | 用新 feature 表征重新训练 weight/responders 多任务头 | 保留用户提出的 feature -> auxiliary -> target 主线，但先提高辅助状态的时序表达能力 |
| P1 | 嵌套时间前向的 auxiliary residual 投影 | 只用 OOF residual 学 responder 组合，避免同段相关性筛选的选择偏差 |
| P1 | 轻量共享编码器蒸馏和 CPU 顺序推理 | 保留多任务表征，同时满足最终 4 核 CPU、12 GB 环境 |
| P2 | 状态专家和更复杂融合 | 只有上述单项 residual 稳定后再增加复杂度 |

## 12. 本轮改动与严格验证结果

### 12.1 Predicted-weight 门控：否决

利用 feature 预测的 weight 做连续/分位状态门控，没有提高已有 auxiliary residual：

- 3 个历史折的门控版本都弱于常数 gamma；
- 最新冻结后半增量从 `+0.00018982` 变成 `-0.00022434`；
- 正增益时间块从 `5/8` 降到 `4/8`；
- `promote_gate_to_test_candidate=false`。

结论：weight 很容易被 feature 预测，不等于它适合作为 target residual 的逐行门控变量。当前不再继续调 gate 形状和温度。

### 12.2 可预测性筛选和 responder 低秩压缩：否决

新增并正式运行了四种结构：

- `predicted_predictable_only`：只保留 aux-valid 相关性不低于 0.5 的辅助变量；
- `predicted_lowrank_only`：按 responder 编号组压缩成低秩状态；
- `raw_plus_predictable_aux`；
- `raw_plus_lowrank_aux`。

历史 3 折仍分别选择旧 `predicted_aux_only`、`predicted_aux_only`、`raw_plus_all_aux`。latest calibration 虽选择 `predicted_predictable_only`，但严格 refit 审计低于现候选：

| 指标 | 现候选 | 可预测性筛选 |
|---|---:|---:|
| 最新 full 增量 | +0.00028620 | +0.00015668 |
| 冻结后半增量 | +0.00018982 | +0.00002504 |
| 正增益时间块 | 5/8 | 5/8 |
| 正增益资产 | 11/15 | 12/15 |
| 正增益 target 幅度桶 | 5/5 | 3/5 |

结论：辅助变量本身难预测，并不代表它对 target 没有条件增量；仅按可预测性删除 responder 会损失信息。

### 12.3 可预测性乘时间外 target 残差相关性：未晋级

在 target-train 前半拟合 raw Ridge，只用后半的时间外 residual 估计每个 predicted auxiliary 的 target 相关性，再乘 aux-valid 可预测性并选择每个辅助模型前 12 个通道。新增结构为：

- 24 维 `predicted_target_relevant_only`；
- 72 维 `raw_plus_target_relevant_aux`。

这两个结构在 4 个折中被 calibration 选中的次数都是 `0/4`。最终四折选择计数：

- `predicted_aux_only`：3/4；
- `raw_plus_all_aux`：1/4；
- 两个 target-relevant 新结构：0/4。

严格审计最终精确复现当前候选的 full `0.00641096`、冻结后半增量 `+0.00018982` 和时间块 `5/8`，但没有任何一项严格超过 incumbent，因此 `all_promotion_checks_passed=false`。这不是新提升，只是回退到原结构。

### 12.4 当前决策

- 保留 `results/auxiliary_stacking_candidate_20260824/submission.zip`；
- 不修改测试候选和 `build_auxiliary_stacking_candidate.py`；
- 不把本地结果表述为线上排行榜提升；
- 下一轮先改 feature 表征和 target 结构，再重新评估 feature -> predicted weight/responders -> target residual 主线。

## 13. 风险和不可确认事项

- 官方没有公开每个 feature 的真实金融含义、公式和窗口；本报告不做虚假命名。
- 官方没有公开 target 的精确计算公式和预测窗口长度。
- 官方没有公开 weight 的精确生成公式；相关性和代理模型不是因果公式。
- responder 由未来区间构造，只能作为训练监督，不能直接成为测试输入。
- 新候选只有本地历史折、最新冻结后半和结构审计证据，没有可核验的线上排行榜提升记录。
- weight 预测相关性高，只说明状态可被 feature 解释；当前 weight-only target 消融并不是最强结构，不能把“weight 可预测”误写成“weight 单独就能提高 target”。
- 最新强基线上的辅助 residual 系数为负，而历史弱 raw control 上为正，说明融合方向依赖 base 模型，不能脱离基线复用 gamma。
- test GPU 批量推理已通过，但最终私榜 Time-Series API 的 4 核 CPU、12 GB 顺序推理仍需单独压缩和验证。

## 14. 文件索引与复现

官方资料：

- [`2026量化交易研究大赛_赛题发布_v1.3.pdf`](2026量化交易研究大赛_赛题发布_v1.3.pdf)
- [`competition_description.md`](../data/raw/public_release_20260630/public_release_20260630/docs/competition_description.md)
- [`data_description.md`](../data/raw/public_release_20260630/public_release_20260630/docs/data_description.md)

本次分析：

- [`analyze_feature_weight_target.py`](../code/tools/analyze_feature_weight_target.py)
- [`analyze_weight_structure.py`](../code/tools/analyze_weight_structure.py)
- [`analysis_summary.json`](../results/feature_weight_target_analysis_20260824/analysis_summary.json)
- [`responder_profile.csv`](../results/feature_weight_target_analysis_20260824/responder_profile.csv)
- [`asset_profile.csv`](../results/feature_weight_target_analysis_20260824/asset_profile.csv)

二阶段模型与审计：

- [`train_auxiliary_stacking.py`](../code/experiments/asset_all_tcn/train_auxiliary_stacking.py)
- [`audit_auxiliary_stacking.py`](../code/experiments/asset_all_tcn/audit_auxiliary_stacking.py)
- [`build_auxiliary_stacking_candidate.py`](../code/experiments/asset_all_tcn/build_auxiliary_stacking_candidate.py)
- [`audit_auxiliary_candidate.py`](../code/experiments/asset_all_tcn/audit_auxiliary_candidate.py)
- [`audit_report.json`](../results/auxiliary_stacking_audit_20260824/audit_report.json)
- [`predicted-weight gate audit`](../results/predicted_weight_gate_20260824/audit_report.json)
- [`predictability/low-rank strict audit`](../results/auxiliary_stacking_reduced_audit_strict_20260824/audit_report.json)
- [`target-relevance strict audit`](../results/auxiliary_stacking_target_relevance_audit_20260824/audit_report.json)
- [`target variant audit table`](../results/auxiliary_stacking_target_relevance_audit_20260824/variant_audit.csv)
- [`candidate metrics.json`](../results/auxiliary_stacking_candidate_20260824/metrics.json)
- [`candidate audit_report.json`](../results/auxiliary_stacking_candidate_20260824/audit_report.json)
- 候选提交：`results/auxiliary_stacking_candidate_20260824/submission.zip`

复现命令：

```bash
conda run -n quant-competition-wsl python -u code/tools/analyze_feature_weight_target.py
conda run -n quant-competition-wsl python -u code/tools/analyze_weight_structure.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_auxiliary_stacking.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/build_auxiliary_stacking_candidate.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_auxiliary_candidate.py
```

## 15. 显式时序输入改进（2026-08-24）

### 15.1 输入已经不再只有当前时点

新增模型在预测 `time_id=t` 时显式读取截至 `t` 的历史窗口：

- 48 个当前稳定 feature 作为低方差控制底座；
- 对排名前 16 个 feature 构造每个 asset 自身的 `lag/delta/rolling`；
- 对排名前 8 个 feature 先聚合同一时点最多 15 个 asset，再构造市场横截面历史；
- lag 为 `1/4/16/32`，rolling 窗口为 `8/32`；
- 时序 Ridge 最终只读取 336 个名称中含 `lag/delta/rolling` 的历史输入，不读取当前横截面列。

正式测试推理维护最近 32 个时点的原始 feature 缓存。测试第一个时点由训练集末尾 32 个时点预热，后续只使用已经出现的测试 feature。lag 还会核验真实 `time_id` 间隔，asset 缺失时点不会被误当成连续历史。

### 15.2 失败方案与最终选择

直接用 LightGBM 学高维时序残差时，calibration 将其权重选择为 `0`；纯时序 LightGBM 信号在冻结后段下降 `-0.00132832`。按 asset 分别拟合 Ridge 虽然全段有正增量，但冻结后段只有 `3/8` 个时间块改善，也未晋级。

最终保留强正则共享 Ridge：先用当前 48 个 feature 的 Ridge 解释 target，再用历史专用 Ridge 学其残差。时序系数只在最早 4,000 个 outer 时点拟合，随后留出 1,000 时点 purge gap，在独立 calibration 和最终 holdout 上冻结验证。

| 区间 | 原候选 | 时序候选 | 增量 |
|---|---:|---:|---:|
| 系数拟合段 | 0.00661364 | 0.00661577 | +0.00000213 |
| 独立 calibration | 0.00537832 | 0.00540041 | +0.00002209 |
| 最后 10,000 时点 holdout | 0.00620254 | 0.00624381 | +0.00004127 |
| outer 全段 | 0.00641096 | 0.00641969 | +0.00000873 |

holdout 为 `5/8` 个时间块改善，冻结时序系数为 `-0.0334517180`。全部本地门槛通过，但全段增量很小，因此状态是“本地时序候选，等待线上验证”，不能宣称已经成为线上最优。

### 15.3 完整测试结果

- 完成 3,217,458 行顺序推理，测试范围 `888480..1105919`；
- CPU 总耗时约 120 秒；
- 两种不同分块大小的 smoke CSV 字节完全一致，结果不依赖分块边界；
- 提交 CSV/ZIP、row_id 顺序、有限值、冻结 gamma 和逐行公式均通过独立审计；
- 原 `auxiliary_stacking_candidate_20260824` 保留不动，新版本单独作为 A/B 候选。

### 15.4 时序候选文件

- [`final_residual_train_predict_ts_features.py`](../code/experiments/asset_all_tcn/final_residual_train_predict_ts_features.py)
- [`audit_temporal_ridge_candidate.py`](../code/experiments/asset_all_tcn/audit_temporal_ridge_candidate.py)
- [`build_temporal_ridge_candidate.py`](../code/experiments/asset_all_tcn/build_temporal_ridge_candidate.py)
- [`audit_temporal_submission.py`](../code/experiments/asset_all_tcn/audit_temporal_submission.py)
- [`local validation audit`](../results/temporal_ridge_candidate_audit_20260824/audit_report.json)
- [`submission audit`](../results/temporal_ridge_candidate_20260824/audit_report.json)
- 候选提交：`results/temporal_ridge_candidate_20260824/submission.zip`
- 模型文件：`models/temporal_ridge_candidate_20260824/temporal_ridge_model.pkl`

复现命令：

```bash
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/final_residual_train_predict_ts_features.py --results-dir results/asset_all_residual_ridge_75k_temporal_history_probe_20260824 --model-dir models/asset_all_residual_ridge_75k_temporal_history_probe_20260824 --skip-test-prediction --ridge-raw-only --residual-model ridge --residual-feature-set historical --residual-ridge-alpha 10000 --top-k 48 --engineered-top-k 16 --market-history-top-k 8 --lag-steps 1 4 16 32 --rolling-windows 8 32 --disable-cross-section --lgbm-n-jobs 8
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_temporal_ridge_candidate.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/build_temporal_ridge_candidate.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_temporal_submission.py
```

尚未完成的是私榜 `main.py/Model.predict()` 的最终 4 核 CPU、12 GB 打包验证。当前构建器已经按相同顺序缓存规则运行，但公共测试提交仍依赖已有 base candidate 的预测文件，不能直接等同于私榜可部署包。
