# Weight 与 Responder 预测方法及 Target 模型增强计划

日期：2026-08-25
状态：本地研究结论，尚无对应线上成绩
范围：使用全部 `feature_*` 预测 `weight` 和 47 个 `responder_*`，再将预测出的辅助状态用于增强 `target`

## 1. 结论摘要

目前已经验证，`feature -> predicted weight/responders -> target` 这条路线是有效的，但不能把所有 responder 简单理解为 target 的替代标签。

核心结论如下：

1. 只使用 323 个 `feature_*`，已经可以较好预测 `weight`，outer 普通 R2 为 `0.81757`；说明 feature 中确实包含较强的活跃度、流动性或交易摩擦代理。
2. 47 个 responder 中，严格 feature-only outer 上有 `38/47` 的普通 R2 为正，`21/47` 达到 `0.5`；但不同 responder 的可预测性差异很大。
3. `responder_31..42` 最容易预测，更像稳定的流动性、摩擦或路径状态轴；`responder_22..27` 和 `44..46` 也较强。
4. 与 target 最相关的 `responder_03/28/02/29` 几乎无法由当前 feature 提前预测。真实 responder 与 target 高相关，不等于测试阶段能够利用。
5. `responder_09..20` 同时具备一定 target 相关性和中等可预测性，是目前最直接的 target 辅助组。
6. `responder_21..27、44..46` 与 target 的普通相关不强，但对 raw target 模型的时间外残差存在条件信息，适合用作残差状态，而不是直接解释为方向标签。
7. 第一版二阶段辅助模型已经把本地强基线从 `0.00612476` 提升到 `0.00641096`，冻结后半增量为 `+0.00018982`，细时间块 `5/8` 改善。
8. 后续显式时序 residual 候选达到 `0.00641969`。因此新的 weight/responder 方案必须超过 `0.00641969`，并在冻结 holdout 和多数时间块上同时改善，才应晋级。

## 2. Weight、Responder 与 Target 的正确关系

### 2.1 Weight

`weight` 是比赛评分中的样本重要性，不表示涨跌方向，也不是 target 的一部分。根据赛题说明，它主要综合未来窗口中的真实成交活跃度、交易摩擦以及样本的相对评估贡献。

一般可理解为：

```text
未来成交更活跃、交易摩擦更低 -> weight 往往更高
```

但官方没有公开精确计算公式，所以不能声称通过若干匿名 feature 恢复了真实 weight。我们预测的是一个可见信息条件下的 `weight proxy`。

它有两种不同用途：

- 训练阶段真实 `weight`：只用作 target 损失的样本权重和官方指标计算。
- 测试阶段 predicted weight：只能作为流动性/重要性状态、门控或 residual 特征，不能替代不存在的真实测试 weight。

### 2.2 Responder

47 个 `responder_*` 是训练阶段可见、由未来窗口构造的辅助响应变量。它们可能覆盖不同窗口下的收益、风险、路径、活跃度、流动性或摩擦等响应轴，但匿名数据无法恢复每一列的真实金融名称。

正确用法是：

```text
当前及历史可见 feature
    -> 预测未来 responder
    -> 得到可在测试阶段生成的辅助状态
    -> 帮助 target 模型或 target residual 模型
```

错误用法是把训练行中的真实 responder 直接作为 target 输入，因为测试阶段没有这些未来变量。

### 2.3 Target

`target` 是未来固定窗口内的风险调整表现类目标。正负号表示方向，绝对值表示未来表现强弱。接近 0 表示未来方向性表现弱，但不表示该样本可以忽略；高 weight 的近零样本若被预测成较大绝对值，同样会受到明显惩罚。

官方分数为 Weighted Zero-Mean R2：

```text
Score = 1 - sum(weight * (target - prediction)^2)
            / sum(weight * target^2)
```

辅助任务报告中的普通 R2 只衡量 `weight/responder` 的可预测性，不等于比赛分数。

## 3. 严格无泄漏时间协议

当前主实验使用以下时间切分：

| 阶段 | time_id | 用途 |
|---|---:|---|
| aux train | `688480..776479` | 用真实 weight/responders 训练辅助模型 |
| aux valid | `777480..787479` | 选择辅助模型、评估辅助变量可预测性 |
| target train | `788480..846479` | 用预测辅助状态训练 target 模型 |
| calibration | `847480..867479` | 选择 target 结构、残差权重和收缩系数 |
| outer | `868480..888479` | 完全时间外验证 |

相邻阶段至少保留 `1000` 个 `time_id` purge gap。

必须满足以下边界：

- 辅助模型只能使用 target 训练段之前的数据拟合。
- target train、calibration 和 outer 中只能使用辅助模型的预测值。
- 真实 responder 绝不能作为 target 模型的行级输入。
- 真实 weight 可以作为训练损失权重和验证评分权重，但测试推理只能使用 predicted weight。
- 所有筛选、PCA、标准化、融合系数和门控阈值都必须在各折训练/calibration 内拟合。

## 4. 当前如何预测 Weight 和 Responders

当前正式实现位于：

- [`train_auxiliary_stacking.py`](../code/experiments/asset_all_tcn/train_auxiliary_stacking.py)
- [`protocol.json`](../results/auxiliary_stacking_20260824/protocol.json)
- [`auxiliary_prediction_metrics.csv`](../results/auxiliary_stacking_20260824/auxiliary_prediction_metrics.csv)

### 4.1 输入和输出

辅助模型的基础输入是全部 323 个 `feature_000..322`。

正式版本额外加入 15 维 `asset_id` one-hot，因此输入共 338 维：

```text
X_aux = [323 standardized features, 15 asset one-hot]
```

模型一次输出 48 个变量：

```text
Z = [weight, responder_00, ..., responder_46]
```

同时保留 feature-only 对照，用来判断模型究竟学到了 feature 状态，还是主要记住了 asset 固定差异。

### 4.2 预处理

所有统计量只在 aux train 上拟合：

1. 对每个 feature 计算均值和标准差。
2. 标准化后将输入裁剪到 `[-10, 10]`。
3. 缺失值和非有限值填为 0。
4. 对 `weight` 使用 `log1p(max(weight, 0))`，降低长尾影响。
5. 48 个辅助标签分别做标准化，避免量纲大的任务支配损失。
6. 推理后对 predicted weight 使用 `expm1` 还原，并截断为非负。

### 4.3 两类辅助模型

多输出 Ridge：

```text
338 -> 48
alpha = 1000
solver = cholesky
```

共享 MLP：

```text
338 -> 256 -> 128 -> 48
GELU + dropout 0.05
Smooth L1 loss
AdamW + CUDA AMP + early stopping
```

当前结果中 Ridge 整体不弱于 MLP，并且跨时间更稳定。因此下一轮应继续把 Ridge 作为必须超过的辅助基线，不能只比较神经网络内部版本。

## 5. Feature-Only 与 Asset-ID 对照

严格 feature-only 对照使用同一时间协议、同一标准化和 Ridge 参数，只删除 15 维 asset one-hot。

| 区间 | 输入 | weight corr | weight 普通 R2 | responder 平均 corr | responder 平均普通 R2 | responder R2 >= 0.5 |
|---|---|---:|---:|---:|---:|---:|
| aux-valid | 323 features | `0.90704` | `0.81213` | `0.54391` | `0.38194` | `16/47` |
| outer | 323 features | `0.91494` | `0.81757` | `0.58466` | `0.42643` | `21/47` |
| aux-valid | features + asset | `0.95698` | `0.91292` | `0.54650` | `0.38925` | `16/47` |
| outer | features + asset | `0.95889` | `0.81611` | `0.59010` | `0.43265` | `21/47` |

解读：

- 删除 asset_id 后，outer responder 平均 R2 只下降约 `0.0062`，说明 responders 主要由 feature 预测，不是简单记住标的编号。
- asset_id 明显提高 predicted weight 的相关性，但 latest outer 的普通 R2 没有提高，说明固定效应有价值，同时也存在跨时间均值或尺度漂移。
- 正式训练仍可保留 asset_id，但应该对 predicted weight 做每折、每时间块和每 asset 的校准审计。
- feature-only 对照的汇总值目前记录在本文；正式结果 CSV 是默认的 feature + asset 版本。下一轮应给训练脚本增加固定的 `--no-asset-id` 开关并单独落盘，避免再次依赖临时对照代码。

## 6. 哪些 Responders 可以预测

按严格 feature-only outer 普通 R2 分组：

| 分组 | responders | 建模判断 |
|---|---|---|
| 很强 | `26、27、31..42` | 可稳定形成辅助状态，其中 `31..42` 主要属于流动性/路径状态候选 |
| 较强 | `22..25、44..46` | 适合作为条件状态或 residual 输入 |
| 中等 | `07..12、14..18、21、43` | 可能包含方向或强度信息，需要强正则和跨折检查 |
| 很弱 | `00、13、19、20` | 只能作为低权重任务，不应单独承担 target 预测 |
| 失败 | `01..06、28..30` | 当前 feature 无法稳定提前预测，默认收缩或只保留在联合低维投影中 |

代表性区间：

- `responder_31..42`：feature-only outer R2 约 `0.75..0.84`。
- `responder_22..27`：R2 约 `0.56..0.72`。
- `responder_44..46`：R2 约 `0.52..0.60`。
- `responder_01..06、28..30`：R2 为负。

这说明不能把 47 个任务等权处理。预测难度不同、金融角色不同、对 target 的用途也不同。

## 7. 哪些 Responders 与 Target 真正有用

### 7.1 直接相关且可以部分预测

下表的 target 相关来自全量描述性分析，预测 R2 来自严格时间外、默认 feature + asset Ridge。相关性只用于理解，不用于 outer 选型。

| responder | corr(true responder, target) | outer 预测普通 R2 | 判断 |
|---|---:|---:|---|
| `18` | `0.4314` | `0.2041` | target 相关较强，可部分预测 |
| `11` | `0.3970` | `0.2165` | 可作为中等强度方向状态 |
| `19` | `0.3905` | `0.1931` | 较弱，需收缩 |
| `17` | `0.3800` | `0.2302` | 可作为方向/状态辅助 |
| `12` | `0.3587` | `0.2038` | 可作为中等强度方向状态 |
| `10` | `0.3572` | `0.2487` | 当前直接辅助组中较有价值 |
| `20` | `0.3033` | `0.1756` | 预测较弱，需收缩 |
| `13` | `0.2840` | `0.1704` | 预测较弱，需收缩 |
| `16` | `0.2475` | `0.2668` | 可预测性相对更好 |
| `09` | `0.2372` | `0.2872` | 可预测性相对更好 |

当前可将 `responder_09..20` 视为“直接 target 组”，但不是每列等权，也不能用全量相关性直接决定模型系数。

### 7.2 与 Target 很相关但无法提前预测

| responder | corr(true responder, target) | outer 预测普通 R2 |
|---|---:|---:|
| `03` | `0.8104` | `-0.0043` |
| `28` | `0.6863` | `-0.0113` |
| `02` | `0.5571` | `-0.0030` |
| `29` | `0.5458` | `-0.0182` |

这四列说明了辅助标签路线中最容易犯的错误：未来 responder 几乎可以描述 target，并不代表当前 feature 已经包含足够信息预测这个未来量。它们可以帮助训练共享表示，但不能根据真实相关性被赋予很高的直接 target 权重。

### 7.3 对 Target 残差有条件信息

当前代码还执行了严格时间外的 residual 筛选：

1. 用 target-train 前半训练 raw Ridge。
2. 在后半计算时间外 target residual。
3. 计算 predicted responder 与 residual 的加权相关。
4. 将相关绝对值乘以该 responder 的可预测性。
5. 只允许辅助可预测性相关不低于 `0.2` 的变量参与。
6. 每折选择前 12 个。

多折较稳定的条件状态是：

- Ridge 至少 3/4 折出现：`21..27、44、46`。
- MLP 至少 3/4 折出现：`22..27、44..46`。
- MLP 4/4 折出现：`24..27`。

这些 responder 与 target 的普通相关接近 0，但可能描述 raw target 模型没有处理好的市场状态。它们更适合进入 residual 模型或 regime gate，而不是直接作为 target 方向代理。

## 8. 已完成的 Target 增强方法

当前有效链路为：

```text
323 features + asset_id
    -> Ridge / MLP 预测 weight + 47 responders
    -> 形成 96 维预测辅助状态
    -> 二阶段 target 模型
    -> 减去同体系 raw auxiliary control
    -> 得到 auxiliary residual delta
    -> 与当前强基线冻结融合
```

最新辅助融合公式是：

```text
auxiliary_delta = two_stage_prediction - raw_auxiliary_control
candidate = current_best - 0.4363287926 * auxiliary_delta
```

`-0.4363` 只表示相对于当前强基线的残差修正方向，不表示 responder 的金融方向为负。

| 区间 | 当前强基线 | 辅助候选 | 增量 |
|---|---:|---:|---:|
| gamma 拟合前半 | `0.00623861` | `0.00662274` | `+0.00038414` |
| gamma 冻结后半 | `0.00601272` | `0.00620254` | `+0.00018982` |
| outer 全段 | `0.00612476` | `0.00641096` | `+0.00028620` |

稳定性：

- 细时间块 `5/8` 改善。
- asset `11/15` 改善。
- `abs(target)` 五分位 `5/5` 改善。
- 最高两个 weight 分桶改善，低 weight 三个分桶下降。
- 这是本地候选，没有线上成绩。

后续显式历史 Ridge residual 在此基础上将 outer 提升到 `0.00641969`，冻结 holdout 从 `0.00620254` 提升到 `0.00624381`。这说明时序信息有效，但当前时序特征主要作用于 target residual，还没有系统加入 weight/responder 辅助头。

## 9. 已验证失败的简单方案

以下方案没有超过完整辅助状态：

1. 只按 responder 自身可预测性设阈值并删除低分任务。
2. 对预定义 responder 分组做简单相关加权低秩压缩。
3. 只保留与 target residual 单折相关的 12 个 responder。
4. `predicted_target_relevant_only` 在 4 个前向折中被 calibration 选中 `0/4`。
5. `raw_plus_target_relevant_aux` 在 4 个前向折中被 calibration 选中 `0/4`。
6. 简单 predicted-weight gate 没有超过完整辅助候选。

可能原因：

- responder 的条件方向会随时间状态翻转。
- 多个 responder 高度共线，单变量排名无法表达联合信息。
- 先按相关性硬删除会丢掉作为控制变量有用、但边际相关较弱的状态。
- 当前辅助输入仍以单时点 feature 为主，无法充分表达 responder 所对应的未来路径条件。

因此下一轮不应继续反复调整单折相关阈值，而应改进时间表征、低维状态学习和 target 头的结构。

## 10. 下一轮模型增强计划

### P0：为辅助头加入显式时序信息

这是当前最重要的缺口。已有历史特征只增强了 target residual，辅助 Ridge/MLP 仍主要读取当前时点输入。

对每个 asset 构造严格因果特征：

```text
lag:      1, 4, 16, 32
delta:    x(t) - x(t-lag)
rolling:  mean/std/min/max/range over 8 and 32
```

对同一 `time_id` 的最多 15 个 asset 构造当前横截面 rank、demean、z-score 和 dispersion，再对市场聚合状态构造历史 lag/rolling。

第一轮只训练强正则 Ridge，比较：

- A0：当前 323 feature-only。
- A1：当前 323 feature + asset_id。
- A2：feature + asset history。
- A3：feature + asset history + market history。

只有 A2/A3 在多数前向折提高 weight 和各 responder 组的时间外 R2，才进入共享 MLP 或小型 TCN。

### P0：按金融角色形成 OOF Latent State

不再把 48 个预测值直接等权交给 target 模型，也不做单折硬删除。按用途建立三组：

| 状态组 | 变量 | 用途 |
|---|---|---|
| 直接 target 状态 | `responder_09..20` | 提供方向、幅度和弱信号条件 |
| 条件 residual 状态 | `responder_21..27、44..46` | 修正 raw target 模型在特定状态下的残差 |
| 流动性/重要性状态 | `weight、responder_31..42` | predicted-weight gate、动态 shrink、高重要性样本状态 |

每组使用多个时间前向折的 OOF predicted responders，压缩为 `2..4` 个 latent state。第一版使用强正则线性投影或 PLS；只有线性版本稳定后再比较小型自编码器/MLP。latent 的选择过程不能访问 outer target。

### P1：Target 使用 residual 增强，不替代现有强模型

保留当前强模型的预测 `p_base`，新辅助模块只学习剩余误差：

```text
r = target - p_base
r_hat = g(raw feature, temporal state, predicted auxiliary latent)
p_new = p_base + gamma * r_hat
```

`gamma` 只能在 calibration 前半拟合，并冻结到后半及测试。这样可以限制辅助模型在方向漂移时破坏强基线。

第一版 `g` 使用 Ridge，第二版再比较浅层 LightGBM 或小型 MLP。所有模型都应同时报告 `gamma=1` 的原始效果和冻结 gamma 后的效果，避免把全部增益归因于事后缩放。

### P1：拆分 Target 的方向、幅度和近零弱信号

建立三个时间前向 OOF 头：

- 方向头：`P(target > 0 | x, z)`。
- 幅度头：正负条件下的 `E[abs(target) | x, z]`。
- 弱信号头：`P(abs(target) <= q | x, z)`，其中阈值 `q` 只在训练折拟合。

辅助状态的使用方式：

- `09..20` latent 主要进入方向和幅度头。
- `weight、31..42` latent 主要控制输出收缩程度。
- `21..27、44..46` latent 主要进入 residual 和 regime gate。

最终连续预测仍以官方加权平方误差优化，不能硬输出正负类别，也不能用真实 `abs(target)` 作为测试置信度。

### P1：处理时间漂移和符号翻转

对每个 latent 在至少 4 个前向折统计：

- 与 target residual 的加权相关符号。
- 最优 gamma 的方向和大小。
- 各时间块、asset、weight 桶中的增量。

如果某 latent 频繁符号翻转，则：

- 使用 predicted liquidity/volatility state 做连续 regime gate；或
- 将该 latent 的融合系数强烈收缩到 0；
- 不允许根据 outer 的符号反向调整。

### P2：部署约束下的序列模型

当因果 Ridge 已证明时序辅助输入有效后，再训练共享 causal TCN：

```text
最近 32 个 time_id
x 15 assets
x 精选 raw/market features
-> shared temporal encoder
-> weight head + responder group heads
```

训练可使用 WSL CUDA，但最终私榜环境只有 4 核 CPU、12 GB 内存、无 GPU。序列模型必须压缩，并通过官方 Time-Series API 顺序推理、状态预热、缺失 asset 和分块一致性测试。

## 11. 实验矩阵

| 实验 | 辅助输入 | 辅助输出 | Target 用法 | 目的 |
|---|---|---|---|---|
| E0 | current feature + asset | 全 48 维 | 当前 residual delta | 固定基线 |
| E1 | feature-only | 全 48 维 | 当前 residual delta | 审计 asset 固定效应 |
| E2 | feature + asset history | 全 48 维 | 当前 residual delta | 检验 asset 时序增量 |
| E3 | E2 + market history | 全 48 维 | 当前 residual delta | 检验 15 标的共同状态 |
| E4 | E3 | 三组 OOF latent | Ridge residual | 降低共线和噪声 |
| E5 | E3 | 三组 OOF latent | 方向/幅度/弱信号头 | 利用 target 结构 |
| E6 | E3 | latent + regime gate | 当前最优模型 residual | 处理符号漂移 |
| E7 | causal TCN | 分组多任务头 | residual | 只在 Ridge 时序有效后进行 |

每个实验必须同时保留：

- 辅助任务：每列和每组的 corr、普通 R2、分时间块 R2。
- target 任务：官方 Weighted Zero-Mean R2。
- 稳定性：前向折、冻结 holdout、细时间块、asset、weight 桶、target 幅度桶。
- 工程性：峰值内存、CPU 推理时间、分块一致性、NaN/Inf 检查。

## 12. 晋级标准

### 12.1 辅助模型晋级

新的 weight/responder 模型不能只看平均 R2。至少满足：

1. weight、直接 target 组、条件 residual 组和流动性组分别报告结果。
2. 多数前向折优于当前 Ridge。
3. latest outer 不能只靠单一 asset 或单一时间段提升。
4. 对很弱 responder 的提升不能以破坏强 responder 为代价。
5. 所有预处理和模型选择均在各折内部完成。

### 12.2 Target 候选晋级

研究候选至少满足：

- calibration 选择完全不查看 outer 后半。
- 冻结 holdout 增量为正。
- 多数历史前向折为正，最差折不出现明显回退。
- 细时间块至少达到当前 `5/8`，优先要求 `6/8`。
- 多数 asset 改善，不能仅依靠高 weight 的一两个标的。

替换当前本地候选还必须：

- outer 全段超过当前本地参考 `0.00641969`。
- 冻结 holdout 超过当前参考 `0.00624381`。
- 完成独立审计和 CPU Time-Series API 全流程。
- 本地结果与线上结果分开记录；没有线上成绩时只能称为本地候选。

## 13. 推荐执行顺序

1. 给辅助训练脚本增加固定的 feature-only 开关，并将对照指标独立落盘。
2. 复用已实现的 lag/delta/rolling 和 15 标的市场历史构造器，接入辅助 Ridge。
3. 完成 E2/E3 多折验证，确认时序信息是否真正提高可预测 responder。
4. 用严格 OOF predicted responders 构造三组 latent state。
5. 先做 Ridge target residual，再做方向/幅度/弱信号头。
6. 对符号不稳定 latent 增加 regime gate，并与当前时序候选融合。
7. 只有 Ridge 时序辅助头通过晋级标准后，才投入共享 MLP/TCN。
8. 最后生成独立候选目录、复算公式、审计提交文件，并等待线上 A/B 验证。

## 14. 文件索引与复现入口

关键研究资料：

- [`Feature、Weight 与 Target 分析报告`](feature_weight_target_analysis_20260824.md)
- [`赛题 PDF v1.3`](2026量化交易研究大赛_赛题发布_v1.3.pdf)
- [`responder_profile.csv`](../results/feature_weight_target_analysis_20260824/responder_profile.csv)
- [`辅助预测指标`](../results/auxiliary_stacking_20260824/auxiliary_prediction_metrics.csv)
- [`辅助实验 summary`](../results/auxiliary_stacking_20260824/summary.json)
- [`target relevance 严格审计`](../results/auxiliary_stacking_target_relevance_audit_20260824/audit_report.json)
- [`target variant 对照`](../results/auxiliary_stacking_target_relevance_audit_20260824/variant_audit.csv)
- [`时序候选审计`](../results/temporal_ridge_candidate_audit_20260824/audit_report.json)

当前正式辅助实验复现命令：

```bash
cd /mnt/e/量化大赛
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/train_auxiliary_stacking.py
```

严格审计与候选构建入口：

```bash
cd /mnt/e/量化大赛
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_auxiliary_stacking.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/build_auxiliary_stacking_candidate.py
conda run --no-capture-output -n quant-competition-wsl python -u code/experiments/asset_all_tcn/audit_auxiliary_candidate.py
```

结论上，下一步不应继续寻找“某几个 responder 直接等于 target”的简单映射，而应把可预测的 responder 视为不同类型的未来状态代理：用时序 feature 提高其可预测性，用严格 OOF latent 降低共线和噪声，再以受约束的 residual 形式增强当前强 target 模型。
