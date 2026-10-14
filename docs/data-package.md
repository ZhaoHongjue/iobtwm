# `iobtwm.data` 参考文档

这份文档是 `src/iobtwm/data/` 的查阅手册：每个文件、每个类、每个函数是干什么的，参数是什么，返回什么。入门和设计取舍请看 [`src/iobtwm/data/README.md`](../src/iobtwm/data/README.md)。代码里的 docstring 是英文的，内容和这里一致。

## 目录

1. [总览](#1-总览)
2. [常用写法](#2-常用写法)
3. [配置参考](#3-配置参考configpy)
4. [batch 里有什么](#4-batch-里有什么)
5. [模块参考](#5-模块参考)
6. [内部数据结构怎么串起来](#6-内部数据结构怎么串起来)
7. [想加新功能改哪里](#7-想加新功能改哪里)
8. [常见问题](#8-常见问题)

## 1. 总览

### 流水线

```
DataConfig
   │
   ▼
manifest.py   load_clip_table      读 manifest，合并成 ClipTable（一行 = 一个节点的一个 2 秒 clip）
   │
   ▼
splits.py     assign_splits        给每个 clip 一个 split 编号：0 train / 1 val / 2 test / -1 丢掉
   │
   ▼
segments.py   build_segments       同录音、同节点、同 split、时间连续的 clip 拼成段 → SegmentIndex
   │
   ▼
store.py      load_store           每段读出来拼成长波形，整段重采样 → SignalStore
              normalize_store      按 train 的统计量缩放（可缓存到磁盘）
   │
   ▼
sampling.py   WindowSampler        每个 epoch 决定从哪些段的哪里切窗口 → (段号, 起点)
   │
   ▼
dataset.py    WindowDataset        按 (段号, 起点) 切出波形，附上标签和距离
spectral.py   SpectralTransform    STFT（在 worker 里或 GPU 上）
   │
   ▼
builder.py    build_data → IoBTData   把上面串起来，每个 split 一个 DataLoader
```

### 文件一览

| 文件 | 一句话 | 主要对外接口 |
|---|---|---|
| `__init__.py` | 包入口，导出常用名字 | 见下面 |
| `config.py` | 所有设置 | `DataConfig`、`SplitConfig`、`SamplingConfig`、`SpectralConfig`、`DEFAULT_SUBSETS` |
| `manifest.py` | 读 manifest | `load_clip_table`、`ClipTable`、`SubsetInfo`、`record_key`、`normalize_label` |
| `splits.py` | 划分 | `assign_splits`、5 个 `split_*`、`allocate_records`、`official_code`、`DROPPED` |
| `segments.py` | 拼段 | `build_segments`、`SegmentIndex` |
| `store.py` | 波形存储 | `load_store`、`normalize_store`、`save_store`、`open_store`、`resample`、`SignalStore` |
| `sampling.py` | 切窗口 | `WindowSampler`、`window_grid`、`plan_windows`、`WindowGrid`、`WindowKey` |
| `dataset.py` | 取样本 | `WindowDataset`、`Example` |
| `spectral.py` | STFT | `SpectralTransform` |
| `builder.py` | 总装 | `build_data`、`IoBTData`、`SplitSummary`、`cache_key`、`tick_rate` |

`from iobtwm.data import ...` 能直接拿到：`DataConfig`、`SplitConfig`、`SamplingConfig`、`SpectralConfig`、`DEFAULT_SUBSETS`、`build_data`、`IoBTData`、`SplitSummary`、`Example`、`SpectralTransform`。

### 两个贯穿全包的概念

- **段（segment）**：同一录音、同一节点、同一 split 里，时间上连续的一串 clip。窗口永远不会跨段，所以不会把两次录音、或者 train 和 test 拼进同一个样本。
- **tick**：时间的最小单位，等于各模态采样率的最大公约数。默认 audio 1600 Hz、seismic 100 Hz，所以 1 tick = 1/100 秒。窗口起点用 tick 表示，这样在每个模态里都正好落在整数个采样点上：1 tick = 16 个 audio 采样点 = 1 个 seismic 采样点。

## 2. 常用写法

### 默认设置训练

```python
from pathlib import Path

from iobtwm.data import DataConfig, build_data

data = build_data(DataConfig(cache_dir=Path("cache/data")))
for epoch in range(100):
    for batch in data.iterate("train", epoch=epoch, device="cuda"):
        loss = model(batch["audio"], batch["seismic"])
    for batch in data.iterate("val", device="cuda"):
        ...
```

默认是：按时间划分（`time`）、8 个 clip 拼成 16 秒窗口（`multi_clip`）、对数幅度谱、在 GPU 上算 STFT。

### 看各 split 有多大

```python
for name, s in data.summary().items():
    print(name, s.segments, f"{s.hours:.1f} h", s.examples)
```

### 换划分方式

```python
from iobtwm.data import SplitConfig

DataConfig(split=SplitConfig(strategy="record"))  # 按录音
DataConfig(split=SplitConfig(strategy="official"))  # M3N/ICT/West Point 官方划分
DataConfig(split=SplitConfig(strategy="holdout", holdout=("GracesQuarters",)))  # 留出 GQ
DataConfig(split=SplitConfig(strategy="holdout", holdout=("M3N-VC/s31",)))  # 留出 M3N 一个场景
DataConfig(split=SplitConfig(strategy="random_clip"))  # 老论文的随机划分
DataConfig(split=SplitConfig(strategy="time", gap_seconds=30.0))  # 时间划分，切口空 30 秒
```

### 换窗口

```python
from iobtwm.data import SamplingConfig

DataConfig(sampling=SamplingConfig(mode="clip"))  # 单个 2 秒 clip
DataConfig(sampling=SamplingConfig(mode="multi_clip", num_clips=12))  # 24 秒，不重叠
# 16 秒，每 2 秒滑一次
DataConfig(sampling=SamplingConfig(mode="multi_clip", num_clips=8, clip_stride=1))
DataConfig(sampling=SamplingConfig(mode="random_crop", window_seconds=24.0))  # 任意起点 24 秒
```

### 换输入表示

```python
from iobtwm.data import SpectralConfig

DataConfig(spectral=SpectralConfig(representation="waveform"))  # 直接给波形
DataConfig(spectral=SpectralConfig(representation="complex"))  # 实部、虚部两个通道
DataConfig(spectral=SpectralConfig(placement="loader"))  # STFT 在 worker 里算
# 复现 FOCAL / cs537aiot 的每 0.2 秒一次 FFT（只保留一半频率）
DataConfig(
    spectral=SpectralConfig(
        representation="complex", window="rect", window_seconds=0.2, hop_seconds=0.2
    )
)
```

### 只用部分数据

```python
from iobtwm.data import DEFAULT_SUBSETS

m3n = tuple(s for s in DEFAULT_SUBSETS if s.startswith("M3N-VC"))
DataConfig(subsets=m3n)  # 只用 M3N-VC
DataConfig(modalities=("seismic",))  # 只要振动
```

### 自己写训练循环

```python
data.set_epoch(epoch)  # 每个 epoch 开始前调一次
for batch in data.loaders["train"]:
    batch = data.prepare(batch, "cuda")  # 搬到 GPU；placement="device" 时顺便做 STFT
```

### 查一个样本是从哪来的

```python
seg = int(batch["segment"][0])
row = data.segments.first_clips()[seg]  # 这段第一个 clip 在 ClipTable 里的行号
t = data.table
print(
    t.datasets[t.dataset[row]],
    t.records[t.record[row]],
    t.nodes[t.node[row]],
    "起点",
    float(batch["start_seconds"][0]),
    "秒",
)
```

### 把多标签变成车型名

```python
names = [data.classes[i] for i in batch["labels"][0].nonzero().flatten().tolist()]
```

## 3. 配置参考（`config.py`）

所有 config 都是 frozen dataclass：创建以后不能改，要改就新建一个，或者用 `dataclasses.replace(cfg, 字段=新值)`。创建时会立刻检查取值，填错直接报 `ValueError`，不会等到读数据时才报。

### `DataConfig`：总设置

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `root` | `Path` | `datasets_unified` | unified 数据的根目录 |
| `subsets` | `tuple[str, ...]` | `DEFAULT_SUBSETS` | 要读的子集，相对 `root` 的路径 |
| `modalities` | `tuple["audio" \| "seismic", ...]` | `("audio", "seismic")` | 样本里返回哪些模态 |
| `audio_rate` | `int` | 1600 | audio 统一重采样到这个采样率（Hz） |
| `seismic_rate` | `int` | 100 | seismic 统一重采样到这个采样率（Hz） |
| `normalization` | `"none" \| "node" \| "dataset"` | `"node"` | 幅度归一化的分组：每个（数据集, 节点）一个系数、每个数据集一个、或者不做 |
| `remove_dc` | `bool` | True | 每个样本、每个模态减掉自己的均值 |
| `split` | `SplitConfig` | 默认值 | 划分设置 |
| `sampling` | `SamplingConfig` | 默认值 | 窗口设置 |
| `spectral` | `SpectralConfig` | 默认值 | 输入表示设置 |
| `batch_size` | `int` | 64 | 所有 loader 的 batch 大小 |
| `num_workers` | `int` | 8 | DataLoader worker 进程数；0 表示在主进程里读 |
| `cache_dir` | `Path \| None` | None | 设置后，重采样和归一化后的波形会缓存在这里 |

方法：`rate(modality) -> int`，返回某个模态的统一采样率。

会报错的情况：`subsets` 为空；`modalities` 为空或有重复；采样率 ≤ 0；`batch_size` < 1；`num_workers` < 0。

### `SplitConfig`：划分

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `strategy` | `"random_clip" \| "time" \| "record" \| "official" \| "holdout"` | `"time"` | 划分方式，见下表 |
| `ratios` | `(float, float, float)` | `(0.8, 0.1, 0.1)` | train、val、test 的比例，加起来要等于 1，train 必须 > 0 |
| `seed` | `int` | 0 | 随机划分的种子，也用作窗口打乱的种子 |
| `gap_seconds` | `float` | 10.0 | 只对 `time` 有效：每个切口后面空出多少秒不用 |
| `stratify` | `tuple["dataset" \| "scene" \| "labels", ...]` | `("dataset", "labels")` | 对 `record`，以及 `official`、`holdout` 里按录音分的那部分有效：在哪些属性相同的组里分录音 |
| `holdout` | `tuple[str, ...]` | `()` | 只对 `holdout` 有效：留出哪些。`strategy="holdout"` 时不能为空 |

| `strategy` | 做法 | 用途 |
|---|---|---|
| `random_clip` | 每个 clip 独立随机分 | 复现老论文；相邻 clip 跨集合，分数虚高 |
| `time` | 每条录音按时间切：前 80% train，接着 10% val，最后 10% test，所有节点在同一时刻切 | 默认；每条录音三个集合都有 |
| `record` | 整条录音分到一边；在 `stratify` 组里打乱后按比例分 | 测换一次录音还行不行 |
| `official` | M3N-VC、ICT 偶数 run 训练、奇数 run 测试；West Point run 0–2、6–8 训练，其余测试；其他数据集退回 `record` | 和论文比数字 |
| `holdout` | `holdout` 匹配到的录音全进 test，其余按录音分 train 和 val | 跨场地泛化 |

`holdout` 的名字可以写成数据集名（`"GracesQuarters"`）、子集路径（`"GracesQuarters/day_2_sensor_all"`），或者「数据集/场景」（`"M3N-VC/s31"`）。

### `SamplingConfig`：窗口

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mode` | `"clip" \| "multi_clip" \| "random_crop"` | `"multi_clip"` | 取样方式 |
| `num_clips` | `int` | 8 | 只对 `multi_clip` 有效：几个 clip 拼一个样本 |
| `clip_stride` | `int \| None` | None | 只对 `multi_clip` 有效：相邻样本起点隔几个 clip；None 表示等于 `num_clips`，即不重叠 |
| `window_seconds` | `float` | 24.0 | 只对 `random_crop` 有效：窗口长度，必须是整数个 tick |
| `eval_stride_seconds` | `float \| None` | None | 只对 `random_crop` 有效：val、test 固定格子的步长；None 表示等于窗口长度 |

| `mode` | 训练集 | val、test |
|---|---|---|
| `clip` | 每个 clip 一个样本，打乱 | 同左，不打乱 |
| `multi_clip` | 从每段开头起，每隔 `clip_stride` 个 clip 取一个窗口，打乱 | 同左，不打乱 |
| `random_crop` | 每个 epoch、每段随机一个偏移，从偏移处起每隔一个窗口长度切一刀，打乱 | 从每段开头起，每隔 `eval_stride_seconds` 切一刀，不打乱 |

### `SpectralConfig`：输入表示

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `representation` | `"waveform" \| "magnitude" \| "log_magnitude" \| "complex"` | `"log_magnitude"` | 波形；幅度；log(幅度 + `log_eps`)；实部和虚部两个通道 |
| `window_seconds` | `float` | 0.2 | STFT 帧长，在每个模态的采样率下都必须是整数个采样点 |
| `hop_seconds` | `float` | 0.1 | STFT 帧移，要求同上 |
| `window` | `"hann" \| "rect"` | `"hann"` | 窗函数 |
| `log_eps` | `float` | 1e-6 | 取 log 前加的小数，防止 log(0) |
| `placement` | `"loader" \| "device"` | `"device"` | STFT 在 DataLoader worker 里算，还是在 `prepare` 时放到 GPU 上算 |

默认设置下每个样本的形状（16 秒窗口）：audio `[1, 161, 159]`，seismic `[1, 11, 159]`，即 `[通道, 频率, 帧]`。

### 类型别名和常量

| 名字 | 是什么 |
|---|---|
| `Modality` | `"audio"` 或 `"seismic"` |
| `SplitName` | `"train"`、`"val"`、`"test"` |
| `SPLIT_NAMES` | `("train", "val", "test")`；下标就是 split 编号 |
| `DEFAULT_SUBSETS` | 默认读的 15 个子集（去重后），排除的理由写在源码注释里 |
| `SplitStrategy`、`StratifyKey`、`SamplingMode`、`Representation`、`WindowFunction`、`SpectralPlacement`、`Normalization` | 上面各字段的可选值 |

## 4. batch 里有什么

`Example` 就是 `dict[str, Tensor]`。DataLoader 拼成 batch 后，每个值前面多一维 B。

| 键 | 类型 | 形状（单个样本） | 说明 |
|---|---|---|---|
| `audio` | float32 | `[采样点]` 或 `[C, F, T]` | 只在 `modalities` 包含它时才有 |
| `seismic` | float32 | 同上 | 同上 |
| `labels` | float32 | `[类别数]` | 多标签 0/1：窗口覆盖到的所有 clip 里出现过的车型；类别名在 `data.classes` |
| `distance` | float32 | `[]` | 窗口内离最近一辆车的最小距离（米），不知道就是 NaN。默认子集里有距离的：M3N-VC（i22 两车场景和 h24 的背景 run 除外）、GQ 2024-08-06 和 08-07、West Point；Parkland、GQ 老数据、GQ 背景录音都没有 |
| `dataset` | int64 | `[]` | 下标，名字是 `data.table.datasets[...]` |
| `segment` | int64 | `[]` | 段号，用来追溯来源，见「常用写法」 |
| `start_seconds` | float32 | `[]` | 窗口在段内的起点（秒） |

`placement="device"` 时，loader 吐出来的 `audio`、`seismic` 是波形，要过一遍 `data.prepare` 或 `data.iterate` 才变成频谱。

## 5. 模块参考

函数签名里 `NDArray` 是 `numpy.typing.NDArray` 的简写。

### 5.1 `config.py`

见[第 3 节](#3-配置参考configpy)。四个 config 类都有 `__post_init__`，负责创建时检查取值。

### 5.2 `manifest.py`：读 manifest

流水线第一步。读每个子集的 `info.json` 和 `manifest.parquet`，去掉重复行，合并成一张表。之后的步骤只用这张表，不会再读 manifest。

去掉的行：
- Parkland 的 `_dwt` 行：波形和对应的普通行逐字节相同，多出来的只是小波系数。
- M3N-VC 的 `o1000` 行（`offset_ms != 0`）：和普通 clip 重叠一半。

#### `class SubsetInfo`

一个子集的基本信息，来自 `info.json`。

| 字段 | 说明 |
|---|---|
| `name: str` | 子集路径 |
| `rates: dict[Modality, float]` | 各模态的原始采样率 |
| `clip_seconds: float` | 一个 clip 多少秒（目前都是 2.0） |

#### `class ClipTable`

所有可用 clip 的大表，一行对应一个（clip, 节点）。字段分两种：
- **名字表**（tuple，排好序）：`subsets`、`datasets`、`records`、`nodes`、`scenes`、`classes`。
- **每行一个值的列**（numpy 数组）：存的是名字表里的下标。比如 `table.datasets[table.dataset[i]]` 就是第 i 行的数据集名。

| 列 | 类型 | 说明 |
|---|---|---|
| `subset` | int32 | 在 `subsets` 里的下标 |
| `row` | int64 | 在该子集 `signals.h5` 里的行号 |
| `dataset`、`record`、`node`、`scene` | int32 | 在对应名字表里的下标；没有场景的数据集，场景名是 `""` |
| `run` | int64 | run 编号；Parkland 这种有录音名、没有 run 编号的是 -1 |
| `clip` | int64 | 在该录音、该节点里是第几个 clip，相邻编号时间上相邻 |
| `clip_start` | float64 | clip 起始时间戳（秒），只有 M3N-VC 有，其他是 NaN |
| `labels` | bool `[行, 类别]` | 多标签矩阵 |
| `distance` | float32 | 离最近一辆有标注车辆的距离（米），没有就是 NaN |

另外还有 `clip_seconds: float`。`len(table)` 返回行数。

#### `load_clip_table(root: Path, subsets: tuple[str, ...]) -> ClipTable`

读取并合并多个子集的 manifest。

报错（`ValueError`）的情况：
- 某个子集不是波形格式（比如 Parkland 的 `individual_samples`）。
- 子集之间 clip 长度不一致。
- 同一个（录音, 节点, clip）出现了两次，说明选的子集里有重复的录音。

#### `read_subset_info(root: Path, subset: str) -> SubsetInfo`

读一个子集的 `info.json`。不是波形格式，或者 audio 和 seismic 的 clip 时长对不上，会报 `ValueError`。

#### `record_key(subset, dataset, recording, run, scene=None) -> str`

生成「这是哪一次录音」的唯一标识。规则：
- 有录音名（Parkland）：`<数据集>/<录音名>`，比如 `Parkland/motor`。Parkland 同一次录音的不同节点分散在 `individual_time_samples` 和 `extra_time_samples` 两个子集里，这样它们会得到同一个 key。
- 有场景（M3N-VC）：`<子集>/<场景>/run<编号>`。
- 其他：`<子集>/run<编号>`，比如 `GracesQuarters/day_1_sensor_all/run3`。

既没有录音名也没有 run 编号时，报 `ValueError`。

#### `normalize_label(label: str) -> str`

标签转小写，并修正已知的拼写错误（`LABEL_ALIASES`：`polairs` → `polaris`，`warhog` → `warthog`）。

#### 常量和内部函数

| 名字 | 作用 |
|---|---|
| `LABEL_ALIASES` | 拼写修正表 |
| `_COLUMNS` | 需要从 manifest 读的列 |
| `_RawRows` | 单个子集过滤后的行，还没转成下标 |
| `_read_rows(root, subset)` | 读一个子集的 manifest，做过滤和标签规范化 |
| `_nearest_distance(distances)` | 多辆车的距离里取最小的已知值 |
| `_encode(values)` | 名字列表 → 排序后的名字表 + 每行下标 |
| `_multi_hot(labels)` | 每行的标签集合 → 类别名表 + 多标签矩阵 |

### 5.3 `splits.py`：划分

流水线第二步。每种划分返回一个 int8 数组，`ClipTable` 每行一个编号：0 train、1 val、2 test，或者 `DROPPED`（-1，不进任何 split）。只有 `time` 会产生 `DROPPED`，就是切口后面空出来的那些 clip。

#### `assign_splits(table: ClipTable, config: SplitConfig) -> NDArray[int8]`

总入口，按 `config.strategy` 调用下面对应的函数。

#### `split_random_clip(n: int, ratios, seed: int) -> NDArray[int8]`

把 n 个 clip 随机打乱，前 `floor(n × train比例)` 个进 train，接着 `floor(n × val比例)` 个进 val，剩下的进 test。

#### `split_time(table, ratios, gap_seconds) -> NDArray[int8]`

每条录音单独按时间切：
- 切点按 clip 编号的范围算（最小编号到最大编号），不是按 clip 个数算。所以录音中间有缺口时，各节点的切点也不会错开。
- 同一录音的所有节点共用切点，同一时刻不会出现在两个 split 里。
- train 在前，val 在中间，test 在最后。val 和 test 开头各空出 `ceil(gap_seconds / clip 时长)` 个 clip，默认 10 秒就是 5 个 clip。
- test 比例为 0 时，取整剩下的零头直接丢掉。

#### `split_record(table, config) -> NDArray[int8]`

整条录音分到一边：所有录音先按 `config.stratify` 分组，组内打乱，再用 `allocate_records` 决定每组几条进 train、val、test。

#### `allocate_records(n: int, ratios) -> tuple[int, int, int]`

决定一组 n 条录音怎么分，返回 (train, val, test) 的条数：
- test 比例 > 0 时至少给一条，保证每组都测得到。
- val 按比例四舍五入，不强制。大部分组只有 2–3 条录音，每组强塞一条进 val 的话，val 会远超比例。
- train 至少留一条，所以只有一条录音的组全进 train。

| n | ratios | 结果 |
|---|---|---|
| 1 | (0.8, 0.1, 0.1) | (1, 0, 0) |
| 2 | (0.8, 0.1, 0.1) | (1, 0, 1) |
| 3 | (0.8, 0.1, 0.1) | (2, 0, 1) |
| 10 | (0.8, 0.1, 0.1) | (8, 1, 1) |

#### `official_code(dataset: str, run: int) -> int | None`

返回某个 run 的官方划分：0（train）或 2（test）。没有官方 run 级划分的数据集（Parkland、GQ）返回 None。

#### `split_official(table, config) -> NDArray[int8]`

有官方划分的录音按 `official_code` 分，其余的按 `record` 的方式分。

#### `split_holdout(table, config) -> NDArray[int8]`

`config.holdout` 匹配到的录音全进 test。其余录音按录音分 train 和 val，比例用 train:val（test 比例在这里不起作用）。

报错（`ValueError`）：某个名字一条录音都没匹配到；或者全部录音都被留出了。

#### `record_names(table, row: int) -> set[str]`

返回 `holdout` 可以用来指代这一行所在录音的名字：数据集名、子集路径；有场景时再加上「数据集/场景」。

#### 常量和内部函数

| 名字 | 作用 |
|---|---|
| `DROPPED` | -1，表示不进任何 split |
| `_WEST_POINT_TRAIN_RUNS` | West Point 官方训练集的 run：{0, 1, 2, 6, 7, 8} |
| `_counts(n, ratios)` | 向下取整算 train、val 的个数 |
| `_first_rows(table)` | 每条录音取一行，用来查这条录音的数据集、run、场景 |
| `_record_groups(table, records, stratify)` | 把录音按分组键分组 |
| `_assign_records(table, records, config, ratios, record_code)` | 组内打乱、按比例分，结果写进 `record_code`；`record`、`official`、`holdout` 共用 |

### 5.4 `segments.py`：拼段

流水线第三步。

#### `class SegmentIndex`

所有段，用「扁平数组 + 偏移」的方式存（CSR 格式）：

| 字段 | 说明 |
|---|---|
| `clips: NDArray[int64]` | 所有段的 clip 依次排开，存的是 `ClipTable` 的行号，段内按时间排序 |
| `offsets: NDArray[int64]` | 长度为段数 + 1；第 i 段是 `clips[offsets[i]:offsets[i+1]]` |
| `split: NDArray[int8]` | 每段的 split 编号 |

| 方法 | 返回 |
|---|---|
| `len(index)` | 段数 |
| `num_clips()` | 每段有几个 clip |
| `first_clips()` | 每段第一个 clip 的行号；段内不变的属性（数据集、录音、节点）可以从这一行查 |

#### `build_segments(table: ClipTable, codes: NDArray[int8]) -> SegmentIndex`

把有 split 编号的 clip 拼成段。按（录音, 节点, clip 编号）排序后，相邻两个 clip 满足下面全部条件才算连在一起：
- 同一子集、同一录音、同一节点、同一 split；
- clip 编号差 1；
- 如果有时间戳（M3N），起始时间正好差一个 clip 时长（误差不超过 1 毫秒，`_START_TOLERANCE_S`）。

### 5.5 `store.py`：波形存储

流水线第四步。

#### `class SignalStore`

所有段的波形，每个模态一个扁平 float32 数组：

| 字段 | 说明 |
|---|---|
| `rates: dict[Modality, int]` | 各模态采样率 |
| `signals: dict[Modality, NDArray[float32]]` | 所有段按顺序拼在一起；从缓存打开时是只读的内存映射 |
| `offsets: dict[Modality, NDArray[int64]]` | 第 i 段是 `signals[m][offsets[m][i]:offsets[m][i+1]]` |

方法：`segment(modality, index)`，返回某一段的波形（视图，改之前要先复制）。

一个重要性质：每段都是整数个 clip，每个 clip 在目标采样率下都是整数个采样点。所以 `signals[m].reshape(-1, clip_samples)` 正好一行一个 clip，行的顺序和 `SegmentIndex.clips` 一致。归一化就是靠这一点做的。

#### `load_store(root, table, segments, rates) -> SignalStore`

读出、拼接、重采样所有段。`rates` 同时决定读哪些模态。

报错（`ValueError`）的情况：
- 一个 clip 在目标采样率下不是整数个采样点。
- 重采样后某段的长度不等于「clip 数 × 每 clip 采样点数」。

#### `normalize_store(store, table, segments, normalization) -> None`

原地缩放：每个 clip 除以它所在组的系数。组按 `normalization` 划分：（数据集, 节点）或者数据集。`"none"` 时什么都不做。

#### `clip_scales(clip_rms, keys, is_train) -> NDArray[float64]`

算每个 clip 的缩放系数，等于它所在组里 **train** clip 的 RMS 中位数：
- 用中位数，是为了不被车开到跟前时那几段特别响的 clip 带偏。
- RMS 为 0 的 clip（没信号）不参与计算。
- 某组在 train 里一个可用 clip 都没有时，用其他组系数的中位数；所有组都没有就用 1。

#### `resample(wave, source: float, target: int) -> NDArray[float32]`

多相滤波重采样（`scipy.signal.resample_poly`，自带抗混叠）。采样率相同时只做类型转换。`source` 不是整数时报 `ValueError`。

#### `save_store(store, directory)` / `open_store(directory) -> SignalStore`

存缓存和读缓存：
- **存**：先写到 `<目录>.tmp`，写完再改名。中途崩溃不会留下写了一半的缓存。
- **读**：用内存映射打开，速度很快；fork 出来的 worker 共享同一份内存，不会各复制一份。

#### 内部函数

| 名字 | 作用 |
|---|---|
| `_read_rows(dataset, rows)` | 读一段需要的 HDF5 行：先读一块覆盖所有行的连续区域，再从里面挑行，比 HDF5 的花式索引快得多 |
| `_load_modality(root, table, segments, modality, rate)` | 读一个模态的所有段：每个 HDF5 文件只打开一次 |
| `_scale_keys(table, clips, normalization)` | 每个 clip 属于哪个归一化组 |

### 5.6 `sampling.py`：切窗口

流水线第五步。

#### `WindowKey = tuple[int, int]`

一个样本的位置：（段号, 段内起点 tick）。

#### `class WindowGrid`

窗口的「格子」，单位都是 tick：

| 字段 | 说明 |
|---|---|
| `window: int` | 窗口长度 |
| `stride: int` | 同一段内相邻窗口起点的间隔 |
| `random_offset: bool` | 每个 epoch 是否给每段的格子加一个 [0, stride) 内的随机偏移 |

#### `window_grid(config, clip_ticks, tick_rate, *, train) -> WindowGrid`

把 `SamplingConfig` 翻译成格子。只有 `random_crop` 区分训练和评估：训练时步长 = 窗口长度，加随机偏移；评估时用固定格子。

#### `plan_windows(lengths, grid, rng) -> (段位置, 起点)`

在每段上铺格子，返回所有放得下的窗口。比窗口短的段跳过。随机偏移的上限会被压低，保证偏移后至少还放得下一个窗口。返回的窗口按段分组、按顺序排列。实现是向量化的，30 万个窗口约 1.4 毫秒。

#### `class WindowSampler(Sampler[WindowKey])`

给 DataLoader 用的采样器，一个 split 一个。

| 方法 | 说明 |
|---|---|
| `__init__(segments, lengths, grid, *, shuffle, seed)` | `segments` 是这个 split 的段号，`lengths` 是这些段的长度（tick）；创建时自动排好 epoch 0 |
| `set_epoch(epoch)` | 排好某个 epoch 的窗口。结果只取决于 (seed, epoch)，同一个 epoch 每次都一样 |
| `len(sampler)` | 当前 epoch 有多少个窗口 |
| `iter(sampler)` | 依次给出当前 epoch 的 `WindowKey` |

采样器在主进程里运行，所以每个 epoch 的变化对常驻 worker（`persistent_workers`）也有效。

#### `to_ticks(seconds, tick_rate, what) -> int`

秒换算成 tick。不是正整数个 tick 时报 `ValueError`。

### 5.7 `dataset.py`：取样本

#### `class WindowDataset(Dataset[Example])`

用 `WindowKey` 当下标的数据集。它没有 `len`，因为每个 epoch 用哪些窗口由采样器决定。

`__init__(store, table, segments, tick_rate, window_ticks, *, remove_dc, transform)`：

| 参数 | 说明 |
|---|---|
| `store` | 波形 |
| `table` | clip 信息，用来取标签和距离 |
| `segments` | 生成 `store` 用的那份段索引 |
| `tick_rate` | tick 的频率 |
| `window_ticks` | 窗口长度（tick） |
| `remove_dc` | 是否减均值 |
| `transform` | 在 worker 里做的 STFT；None 表示返回波形 |

| 方法 | 说明 |
|---|---|
| `dataset[(段号, 起点)]` | 返回一个 `Example` |
| `_wave(modality, segment, start)` | 复制出一个模态的窗口。用 `np.array` 复制，所以不会改动缓存里的只读数据 |
| `_targets(segment, start)` | 窗口碰到的 clip 的标签并集和最小距离 |

#### `Example`

`dict[str, Tensor]`，内容见[第 4 节](#4-batch-里有什么)。

### 5.8 `spectral.py`：STFT

#### `class SpectralTransform(nn.Module)`

| 方法 | 说明 |
|---|---|
| `__init__(config, rates)` | 为每个模态准备窗函数。帧长或帧移不是整数个采样点时报 `ValueError` |
| `forward(inputs)` | 对 dict 里所有模态做变换，其他键原样保留，返回一个新 dict |
| `transform(modality, wave)` | 变换单个模态：`[..., 时间]` → `[..., C, F, T]`；`waveform` 时原样返回 |
| `output_shape(modality, num_samples)` | 不用真算，就能知道输出形状 |

属性：
- `config`：`SpectralConfig`。
- `frames`：每个模态的（FFT 长度, 帧移），单位是采样点。

其他细节：
- 窗函数注册为不保存的 buffer：`.to(device)` 时跟着走，但不会存进 checkpoint。
- `center=False`：每一帧只用窗口内部的采样点，不在两头补零。
- 频率维是 `n_fft // 2 + 1`，只保留不重复的那一半频谱。
- 帧数是 `(采样点数 - n_fft) // 帧移 + 1`。

`_frame_length(seconds, rate, what)`：内部函数，把秒换算成采样点，不是正整数时报错。

### 5.9 `builder.py`：总装

#### `build_data(config: DataConfig) -> IoBTData`

跑完整条流水线。步骤：
1. 先建 `SpectralTransform`，提前检查 STFT 参数，免得读了几十秒数据才报错。
2. 读 manifest。
3. 划分。
4. 拼段。
5. 读波形或打开缓存。
6. 每个 split 建采样器和 DataLoader。

没有样本的 split 不建 loader。如果是有段、但没有一段长到放得下一个窗口，会打一条 warning。

#### `class IoBTData`

| 属性 | 说明 |
|---|---|
| `config` | 构建时用的设置 |
| `table` | `ClipTable` |
| `segments` | `SegmentIndex` |
| `store` | `SignalStore` |
| `transform` | `SpectralTransform` |
| `samplers` | `dict[split 名, WindowSampler]` |
| `loaders` | `dict[split 名, DataLoader]`，没有样本的 split 不在里面 |
| `classes` | 类别名，就是 `table.classes` |

| 方法 | 说明 |
|---|---|
| `iterate(split, *, epoch=0, device="cpu")` | 排好这个 epoch 的窗口，然后逐个给出处理好的 batch；split 没有 loader 时报 `KeyError` |
| `set_epoch(epoch)` | 所有 split 都排好这个 epoch |
| `prepare(batch, device)` | 搬到 `device`；`placement="device"` 时顺便做 STFT |
| `summary()` | 每个 split 的段数、小时数、每个 epoch 的样本数（`SplitSummary`），空的 split 也会列出 |

#### `class SplitSummary`

字段：
- `segments`：段数。
- `hours`：小时数，所有节点加起来。
- `examples`：每个 epoch 的样本数。

#### `cache_key(config) -> str`

缓存目录名，是对影响波形内容的设置做哈希：数据根目录、子集、采样率、归一化、划分设置，以及缓存版本号 `_CACHE_VERSION`。窗口和 STFT 设置不在里面，所以改这些会直接复用缓存。

#### `tick_rate(config) -> int`

各模态采样率的最大公约数。

#### 内部函数

| 名字 | 作用 |
|---|---|
| `_store(config, table, segments)` | 有缓存就打开，没有就读数据、归一化、存缓存，再用内存映射打开 |
| `_loader(config, dataset, sampler)` | 建 DataLoader，worker 用 fork 方式启动，共享内存里的波形 |
| `_split_loader(...)` | 建一个 split 的采样器和 loader |

## 6. 内部数据结构怎么串起来

```
ClipTable（每行一个 clip）
   row ──► signals.h5 的行号
   record / node / clip ... ──► 名字表

SegmentIndex
   clips   = [ClipTable 行号, 行号, ...]   ← 段 0 的 clip | 段 1 的 clip | ...
   offsets = [0, n0, n0+n1, ...]

SignalStore（每个模态一份）
   signals = [段 0 的波形 | 段 1 的波形 | ...]   每段长度 = clip 数 × 每 clip 采样点数
   offsets = [0, ..., ...]

WindowKey = (段号, 起点 tick)
   段号 ──► SignalStore 里的哪一段，SegmentIndex 里的哪些 clip（取标签）
   起点 × (采样率 / tick_rate) ──► 段内第几个采样点
```

一个具体例子（默认设置）。窗口 `(段 12, 起点 400)`：
- audio 从第 12 段的第 400 × 16 = 6400 个采样点开始，取 16 秒 × 1600 = 25600 个采样点。
- seismic 从第 400 个采样点开始，取 1600 个。
- 标签取第 12 段里第 400 // 200 = 2 个到第 (400 + 1600) / 200 - 1 = 9 个 clip 的并集。

## 7. 想加新功能改哪里

| 想做的 | 改哪里 |
|---|---|
| 加一种划分 | `config.py` 的 `SplitStrategy` 加个名字，`splits.py` 写一个 `split_xxx` 并在 `assign_splits` 里加一个 `case`；按录音分的可以复用 `_assign_records` |
| 加一种取样方式 | `config.py` 的 `SamplingMode`，`sampling.py` 的 `window_grid` |
| 加一种输入表示（比如 log-mel） | `config.py` 的 `Representation`，`spectral.py` 的 `transform` 和 `output_shape` |
| 样本里多返回一个东西（比如节点坐标） | `manifest.py` 把列读进 `ClipTable`，`dataset.py` 的 `__getitem__` 里加上 |
| 加新的数据集或子集 | 先用 `scripts/preprocess` 转成 unified 格式，再加进 `DataConfig.subsets` 或 `DEFAULT_SUBSETS`；有官方划分的话在 `official_code` 里加规则 |
| 改了存储内容的计算方式 | 把 `builder.py` 里的 `_CACHE_VERSION` 加 1，旧缓存就不会再被用到 |
| 多节点同步样本 | 见 README 最后一节「以后要做」 |

## 8. 常见问题

**改了窗口或 STFT 设置，要重建缓存吗？**
不用。缓存只存重采样和归一化后的波形，这些设置不影响它。改了子集、采样率、归一化方式或划分设置，才会生成新的缓存。

**缓存放在哪、占多大？**
在 `cache_dir/<16 位哈希>/` 下，每个模态一个 `.npy` 文件。默认设置约 4.6 GB。可以随便删，下次会自动重建（约 45 秒）。

**为什么 `loaders` 里少了 val 或 test？**
那个 split 一个样本都没有。常见原因：用了 `random_clip` 划分，又用了 `multi_clip` 或 `random_crop`，录音被切得太碎，拼不出窗口。日志里会有 warning。

**为什么 `dataset[i]` 用整数下标会报错？**
`WindowDataset` 的下标是 `(段号, 起点 tick)`，不是整数。要遍历就用 loader，或者 `for key in data.samplers["train"]: data.loaders["train"].dataset[key]`。

**训练结果能复现吗？**
能。划分取决于 `SplitConfig.seed`，每个 epoch 的窗口取决于 (seed, epoch)。模型初始化和 dropout 的随机性要你自己设种子。

**num_workers 设多少？**
8 个 worker 时每秒能出几千个样本，一般够用。`placement="loader"` 时 STFT 在 worker 里算，CPU 压力大一些；`placement="device"` 时 worker 只切波形，很轻。

**各数据集的数据量差很多，要平衡吗？**
目前没做。M3N-VC 约占 60%。需要的话可以在 `WindowSampler` 里加按数据集加权的采样。
