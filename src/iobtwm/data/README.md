# `iobtwm.data`：训练用的数据加载

这个包把 `datasets_unified/` 里的数据变成训练时直接能用的 DataLoader。所有能调的东西都是 `DataConfig` 里的一个字段，换设置不用改代码。

每个类、函数、参数的详细说明见参考文档 [`docs/data-package.md`](../../../docs/data-package.md)。

## 怎么用

```python
from pathlib import Path

from iobtwm.data import DataConfig, SamplingConfig, SpectralConfig, SplitConfig, build_data

data = build_data(
    DataConfig(
        split=SplitConfig(strategy="time"),
        sampling=SamplingConfig(mode="multi_clip", num_clips=8),
        spectral=SpectralConfig(representation="log_magnitude", placement="device"),
        cache_dir=Path("cache/data"),
    )
)
print(data.summary())  # 每个 split 多少段、多少小时、一个 epoch 多少样本

for epoch in range(100):
    for batch in data.iterate("train", epoch=epoch, device="cuda"):
        batch["audio"]  # [B, C, 频率, 帧]，或 representation="waveform" 时 [B, 采样点]
        batch["seismic"]
        batch["labels"]  # [B, 类别数]，多标签；类别名在 data.classes
        batch["distance"]  # [B]，离最近那辆车多少米，不知道就是 NaN
```

`iterate` 每个 epoch 会重新排窗口（随机裁切时换偏移），然后把 batch 搬到 `device` 上。需要的话，FFT 也在那里算。想自己写循环，就用 `data.set_epoch(epoch)`、`data.loaders[split]` 和 `data.prepare(batch, device)`。

## 数据流

1. **读 manifest**（`manifest.py`）：把选中的子集合并成一张 clip 表，一行对应一个（clip, 节点）。会去掉 `_dwt` 重复行和 M3N 的 `o1000` 偏移 clip，统一标签拼写（`warhog` → `warthog`，`polairs` → `polaris`），并给每行算一个 record key。同一次录音的不同节点共用一个 key；Parkland 的 individual 和 extra 里同名的录音也算同一次。
2. **划分**（`splits.py`）：给每个 clip 分配 train、val、test，或者丢掉。
3. **拼段**（`segments.py`）：同一录音、同一节点、同一 split，clip 编号连续的拼成一段。有 `clip_start` 的（M3N）还要求起始时间正好差 2 秒。
4. **读波形**（`store.py`）：每一段从 HDF5 读出来，拼成一条长波形，整段一起重采样到统一采样率，然后按训练集算的尺度归一化。结果可以缓存到 `cache_dir`，下次直接用内存映射打开。
5. **切窗口**（`sampling.py`）：决定这个 epoch 用哪些段、从哪里切。
6. **取样本**（`dataset.py`）：按窗口取出波形，减均值，可选做 FFT（`spectral.py`），再附上标签和距离。

## 选项

### 划分：`SplitConfig.strategy`

| 取值 | 意思 | 什么时候用 |
|---|---|---|
| `random_clip` | 每个 clip 各自随机分，早期组里论文在 MOD 上就是这么做的 | 只用来复现老论文的数字。相邻 clip 会跨 split，分数虚高 |
| `time`（默认） | 每条录音按时间切：前 80% train、中间 10% val、最后 10% test，所有节点在同一时刻切，切口后面空出 `gap_seconds`（默认 10 秒） | 每个数据集、每条录音在三个集合里都有。test 是同一次录音的另一段时间，比按录音分宽松 |
| `record` | 整条录音分到一边，在 `stratify` 指定的组里随机分 | 测换一次录音还行不行，比 `time` 严格 |
| `official` | M3N-VC、ICT、West Point 用官方的 run 划分，其他数据集退回 `record` | 要和 SPAR、RestoreML 这些论文比数字的时候 |
| `holdout` | `holdout` 里列出的数据集、子集或场景整个拿去当 test，剩下的按录音分 train 和 val | 跨场地泛化测试 |

细节：
- `official`：M3N-VC 和 ICT 偶数 run 训练、奇数 run 测试。West Point 用 run 0–2、6–8 训练，其余测试，和 `splits/2026-04-14-West-Point/indices` 一致。ICT 和 West Point 官方把留出的部分叫 val，这里统一当 test。这三个数据集官方都没有 val，所以 val 只能从 Parkland、GQ 里来，会很小。
- `holdout` 的名字有三种写法：数据集名（`GracesQuarters`）、子集路径（`GracesQuarters/day_2_sensor_all`）、或「数据集/场景」（`M3N-VC/s31`）。写错的名字会直接报错。test 的比例在这里不起作用，只用 train:val 的比例。
- `record`：每组至少分一条录音去 test，保证每种车都测得到；val 按比例四舍五入，不强制。只有一条录音的组（Parkland 的 tesla、mustang）只能进 train，会打日志。`stratify` 默认按（数据集、车型）分组。加上 `scene` 能保证 M3N 每个场景两边都有，但大部分组只剩两条录音，只能 1:1 地分。

默认子集上各种划分的大小（小时，所有节点加起来）：

| 划分 | train | val | test |
|---|---|---|---|
| `record` | 131.0 | 10.2 | 46.8 |
| `record`，`stratify` 加 `scene` | 106.7 | 2.5 | 78.8 |
| `official` | 125.6 | 0.4 | 62.1 |
| `holdout=("GracesQuarters",)` | 144.2 | 8.5 | 35.4 |
| `holdout=("M3N-VC/s31",)` | 157.1 | 17.1 | 13.9 |
| `time`（默认） | 151.4 | 16.6 | 16.6 |
| `random_clip` | 150.5 | 18.8 | 18.8 |

### 取样：`SamplingConfig.mode`

| 取值 | 一个样本是什么 |
|---|---|
| `clip` | 一个原始 2 秒 clip |
| `multi_clip`（默认） | `num_clips`（默认 8，即 16 秒）个连续 clip 拼起来；`clip_stride` 是相邻样本起点隔几个 clip，默认不重叠，和 FOCAL 的 sequence sampler 一样 |
| `random_crop` | 长度 `window_seconds`，起点任意。训练时每个 epoch 每段用「格子 + 随机偏移」切，每个时刻每个 epoch 大约用到一次；val 和 test 用固定格子，步长 `eval_stride_seconds` |

比窗口还短的段会被跳过。`random_clip` 划分会把录音切得很碎，配 `multi_clip` 或 `random_crop` 时 val、test 基本没有样本，这时会打 warning，那两个 split 也不会有 loader。

### 输入：`SpectralConfig`

| 字段 | 取值 |
|---|---|
| `representation` | `waveform`（不变换）、`magnitude`（\|X\|）、`log_magnitude`（log(\|X\| + `log_eps`)，默认）、`complex`（实部和虚部两个通道） |
| `placement` | `device`（默认）：loader 给波形，`prepare` 在 GPU 上算 STFT。`loader`：在 DataLoader 的 worker 里算，拿到的直接是频谱 |
| `window_seconds`、`hop_seconds`、`window` | 默认 0.2 秒帧长、0.1 秒帧移、Hann 窗 |

FFT 不预先存盘，存的是波形，训练时再算。这样改 STFT 参数不用重新生成数据。两种 `placement` 的结果完全一样。`window="rect"`、帧长和帧移都设 0.2 秒、`representation="complex"`，就等价于 FOCAL 和 cs537aiot 每 0.2 秒做一次 FFT。区别是这里只保留不重复的一半频率。

### 其他（`DataConfig`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `subsets` | `DEFAULT_SUBSETS` | 去重后的波形子集，排除的理由写在 `config.py` 里 |
| `audio_rate`、`seismic_rate` | 1600、100 Hz | 所有数据集统一重采样到这个采样率 |
| `normalization` | `node` | 每个（数据集, 节点）除以训练集 clip RMS 的中位数；也可以 `dataset` 或 `none` |
| `remove_dc` | True | 每个样本每个模态减掉自己的均值 |
| `cache_dir` | None | 设置以后，第一次构建会存一份，之后内存映射打开 |

## 实测（全部默认子集）

- 第一次构建约 45 秒，读缓存约 3 秒。
- 波形 4.6 GB（audio 4.3 GB，seismic 0.3 GB），构建时峰值内存约 18 GB。
- 8 个 worker、batch 64、24 秒窗口时，每秒能取几千个样本，不会卡训练。
- 我抽查过：没经过重采样的 clip，取出来和 HDF5 里的原始数据逐点一致。

## 已知限制

- GQ、ICT、West Point 相邻 clip 之间是不是真的无缝，还没验证过。M3N 有时间戳，Parkland 从代码上看是无缝切的，这两个没问题。在验证之前，`random_crop` 在这三个数据集上切到的窗口可能跨过接缝。
- 比窗口还短的段直接丢掉，没有补零加 mask。
- `labels` 是窗口覆盖到的所有 clip 的车型并集，`distance` 是这些 clip 里的最小值。还没有逐帧的标签。
- 一个子集里所有样本的采样率必须一样，而且不支持非整数采样率。

## 以后要做：多节点同步样本

现在一个样本只来自一个节点。但一辆车开过去的时候，周围 6–8 个节点是同时在录的：离得近的听得响，离得远的听得轻，声音到达也有先后。多节点同步样本的意思是：一个样本 = 同一段时间、所有节点的录音放在一起。形状从 `[T]` 变成 `[节点数, T]`，M3N 还能附上每个节点的坐标（`node_xy`）。

**为什么要做**：单节点的 world model 只能学到「车越来越响」或「越来越轻」。多节点的能学到「车在哪、往哪开」，可以预测车下一刻在哪。SPAR 就是这么做的，多车检测比基线高了很多。

**要解决的问题**：
- **时间对齐**：M3N 每个 clip 有 `clip_start`，按时间戳对齐就行。GQ、ICT、West Point 只能假设「同一 run、同一 clip 编号 = 同一时刻」，要先验证。Parkland 同一录音不同节点的 clip 数不一样（看起来各自裁剪过），编号可能对不上，要单独处理。
- **节点数不一样**：M3N 各场景 5–8 个节点，还有节点缺数据的情况，需要补零加 mask，或者只用齐全的时间段。
- **划分**：同一时刻的所有节点必须在同一个 split。`record`、`official`、`holdout` 本来就按整条录音分，没问题；`random_clip` 和 `time` 要改成按时刻分。
- **数据结构**：拼段要从（录音, 节点）改成（录音, 时间段），每段存所有节点。sampler 不用大改，dataset 要返回 `[节点数, T]` 和 mask。
- **模型**：要能处理一组节点，还要用到节点坐标，可以参考 SPAR 的做法。

建议等单节点的 world model 训通了再做。
