# 私榜策略文件与 Time-Series API 要求

日期：2026-08-25

本文把官网当前前端、公开赛题说明和本地 runner 的要求整理成最终策略包检查单。官网与飞书的登录后补充规则可能变化；包大小、预装依赖和账号剩余额度以登录后的官方页面为准。

## 1. 最终交付物

私榜提交的是一个策略压缩包，不是公榜预测 CSV。

```text
final_strategy.zip
├── main.py                 # 必须位于压缩包根目录
├── model/                  # 预训练参数、模型文件、标准化参数
├── utils.py                # 如有需要
└── README.md               # 可选，写版本和复现信息
```

压缩包根目录必须直接存在 `main.py`。现有 `Quant-main.zip` 的入口位于 `Quant-main/xgb_strategy/main.py`，直接上传会因根目录缺少 `main.py` 而不合格。现有 `submission.zip` 只有 `submission.csv`，属于公榜交付物，也不能替代策略包。

官网前端目前接受 `.zip,.rar,.7z`，建议只上传 `.zip`，并在上传前记录 SHA256。每队策略文件累计最多 10 次，最终采用最新成功提交版本；截止日期为 2026-08-31 23:59（UTC+8）。

## 2. `main.py` 接口

```python
class Model:
    def __init__(self):
        # 只执行一次：加载轻量模型、参数和必要的训练末尾历史状态
        pass

    def predict(self, test):
        # 每次接收一个 time_id 的当前截面
        # 返回与 test 行数相同的一维有限数值数组
        pass
```

官方语义是：先初始化一次 `Model()`，之后按严格递增的 `time_id` 调用 `predict(test)`。`test` 只包含：

```text
row_id,time_id,asset_id,feature_*
```

`test` 不包含 `target`、`weight`、`responder_*`、真实时间戳、真实 symbol 或未来测试数据。模型可以在自身状态中保留过去已经看到的 feature。

## 3. 返回值与 15 个标的

`predict(test)` 返回的数量必须是 `len(test)`，并且顺序必须与输入行完全一致。通常一个时点有 15 个标的，因此通常返回 15 个 target；但是个别时点可能缺少标的，此时可能是 13、14 或其他实际行数，不能硬编码为 15。

```text
time_id=t      -> test 有 15 行 -> 返回 15 个预测
time_id=t+1    -> test 有 14 行 -> 返回 14 个预测
```

官方 runner 会把每个时点的输入 `row_id` 与返回值合并成最终的：

```text
row_id,target
```

策略代码不需要自己生成最终 CSV。异常、超时、返回长度错误、NaN/Inf 会使相应时点预测被置为 0；初始化失败或严重资源违规可能导致整份提交无效。

## 4. 历史窗口

规则没有规定固定的最大历史时间点数，实际限制是内存、速度、因果性和官方资源。当前项目已经验证过的时序候选使用最近 32 个 `time_id`：

```text
历史缓存：32 个 time_id
lag：1、4、16、32
rolling：8、32
```

预测 `time_id=t` 时只能使用 `t` 和已经出现的历史。当前时点应先计算预测，再写入历史缓存；不能先写入再计算，以免把当前数据错误地当成过去。

```python
from collections import deque

class Model:
    def __init__(self):
        self.history = deque(maxlen=32)
        self.last_time_id = None

    def predict(self, test):
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must increase")

        prediction = self.predict_from_current_and_history(test, self.history)
        self.history.append(test.copy())
        self.last_time_id = time_id
        return prediction
```

历史最好按 `asset_id` 保存，并检查真实 `time_id` 间隔；缺失 asset 不能被当成连续观测。窗口可以扩大到 64 或 128，但必须先通过严格时间前推验证和官方 runner；不能因为窗口更长就假定分数更高。禁止保存或读取全部未来测试数据。

## 5. 资源和依赖

最终环境：4 核 CPU、12 GB RAM、无 GPU、无外部网络。模型初始化上限 180 秒，平均单个 `time_id` 推理目标约 50 ms。最终代码应：

- 将 LightGBM/XGBoost/PyTorch 等线程显式限制为不超过 4；
- 不依赖训练机路径、Windows 路径、网络下载或运行时安装包；
- 只把官方环境可用的依赖和模型文件放进 ZIP；
- 避免在每次 `predict()` 中重复加载模型或扫描整份数据；
- 对缺失值做确定性处理，保证输出有限。

本地 runner 应重点查看：`model_init_seconds`、`predict_total_seconds`、`predict_calls`、`mean_predict_seconds`、`max_predict_seconds`、`predict_timeout_count`、`total_seconds`。

示例命令：

```bash
cd /mnt/e/量化大赛
conda run --no-capture-output -n quant-competition-wsl python data/raw/public_release_20260630/public_release_20260630/timeseries_api/run_timeseries_api.py --data-root data/raw/public_release_20260630/public_release_20260630/data --strategy-dir path/to/strategy --output /tmp/strategy_submission.csv --model-init-timeout-seconds 180 --per-step-timeout-seconds 0.05 --timeout-policy zero_remaining
```

## 6. 提交前检查单

- [ ] ZIP 根目录有 `main.py`，且定义 `Model`。
- [ ] `Model()` 只加载本地随包文件，不访问网络。
- [ ] `predict(test)` 按输入行顺序返回 `len(test)` 个有限数值。
- [ ] 没有读取 `target`、`weight`、`responder_*` 或未来 test。
- [ ] 历史缓存只保留已经到达的时间点；当前时点在预测后写入。
- [ ] 允许缺失 asset，不能假设每个时点必有 15 行。
- [ ] 线程数显式限制为不超过 4。
- [ ] 小样本、缺失 asset、NaN/Inf、重复/倒序 time_id 烟测通过。
- [ ] 完整测试集 runner 状态为 `ok`，无超时和异常消息。
- [ ] 分块大小改变时输出一致，row_id 顺序正确。
- [ ] 记录 ZIP 内容、版本、依赖、模型文件和 SHA256。
- [ ] 登录官网确认策略剩余额度，上传后保存成功页面或截图。

## 7. 当前项目结论

当前本地最佳时序候选约为 `0.0064196889`，其公榜 CSV/ZIP 和离线顺序推理已经审计通过，但仍依赖提前生成的 base prediction，尚未完成最终私榜 `main.py/Model.predict()` 的 4 核 CPU 打包验证。因此当前优先级是把候选改造成实时策略并跑官方 runner，而不是把现有 `submission.zip` 直接上传。

参考：

- [competition_description.md](../competition_description.md)
- [Time-Series API README](../data/raw/public_release_20260630/public_release_20260630/timeseries_api/README.md)
- [feature_weight_target_analysis_20260824.md](feature_weight_target_analysis_20260824.md)
- [官网](https://race.xhth.cn/)
