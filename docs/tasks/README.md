# 数据集总览

这个目录给 `datasets/` 里的每个数据集写了一份说明：数据是什么、能做什么任务、原始文件和 unified 版本各在哪、有哪些坑。

| 数据集 | 说明文档 | 一句话 | 样本数（unified） |
|---|---|---|---|
| Parkland | [Parkland/README.md](Parkland/README.md) | 公园/停车场里单车（和行人）经过传感器，11 类，最老的一批数据 | 78,521 |
| GracesQuarters | [GracesQuarters/README.md](GracesQuarters/README.md) | 野外试验场 Graces Quarters 的越野车/无人车数据：单车、多车、背景、合成混合，文件夹之间大量重复 | 143,987 |
| ICT | [ICT/README.md](ICT/README.md) | 2024-03-29 的四辆民用车数据（带距离），外加合成两车混合 | 61,870 |
| WestPoint | [WestPoint/README.md](WestPoint/README.md) | 2026-04-14 在 West Point 采的两辆车，带距离 | 28,804 |
| M3N-VC | [M3N-VC/README.md](M3N-VC/README.md) | 公开数据集 M3N-VC，6 个场景，位置/距离/时间戳最全 | 412,564 |
| **合计** | | 32 个 subset | **725,746** |

## 所有数据的共同点

- **两个模态**：每个样本都是同一时刻、同一个传感器节点上的一段 **audio（麦克风）** 和一段 **seismic（地震波/地面振动，geophone）**。
- **一个样本 = 一个短 clip**：绝大多数是 2 秒，`ict_pt_files` 是 4 秒。clip 是从一次连续录音（run）里按时间顺序切出来的，同一个 run 里 `clip` 编号相邻就是时间上相邻。
- **核心任务是车辆分类**：听声音 + 感受地面振动，判断经过的是哪种车（或者没车 = background）。部分数据集还有每个 clip 的车-传感器 **距离**（米），可以做距离回归；还有多车同时出现的数据，可以做多标签分类。
- **采样率只有两种组合**：
  - Parkland / GracesQuarters / ICT / WestPoint：audio 8000 Hz，seismic 100 Hz，2 秒 = 16000 + 200 个点。
  - M3N-VC：audio 1600 Hz，seismic 200 Hz，2 秒 = 3200 + 400 个点。
- **节点名**：`rs1`、`rs2`… 是传感器节点编号（原始数据里位置字段叫 `shake`，应该是 Raspberry Shake 一类的设备）；GracesQuarters 用 `gq-1`… 命名。不同数据集里同名节点不是同一个位置。
- **原始文件格式**：每个样本一个 `.pt` 文件（`torch.load` 读），里面是一个 dict，波形放在 `data[<node 或 "shake">][<"audio"|"seismic">]`。

## 数据放在哪

两份数据都在这台机器上，都被 `.gitignore` 忽略，不进 git。

| | 路径 | 大小 | 说明 |
|---|---|---|---|
| 原始数据 | `/data/hongjue2/iobtwm/datasets/` | 约 69 GB | 只读，不要改 |
| Unified 版本 | `/data/hongjue2/iobtwm/datasets_unified/` | 约 46 GB | `scripts/preprocess/convert.py` 生成 |

Unified 版本的规则很简单：**原始的每个样本文件夹 = 一个 subset，路径一模一样**。比如原始的 `datasets/M3N-VC/a06/individual_time_samples/` 对应 `datasets_unified/M3N-VC/a06/individual_time_samples/`。每个 subset 文件夹里有三个文件：

| 文件 | 内容 |
|---|---|
| `signals.h5` | 波形。`signals/audio` 和 `signals/seismic` 都是 `[N, L]` 的二维数组，第 `i` 行就是第 `i` 个样本 |
| `manifest.parquet` | 每行一个样本的元数据：`sample_id`、`labels`、`distances_m`、`node`、`run`、`clip`、原始文件路径和 SHA-256、完整原始元数据 `original_meta` 等 |
| `info.json` | 这个 subset 的摘要：样本数、采样率、shape/dtype、标签和节点计数、注意事项 |

另外 `datasets_unified/` 根目录还有：

- `catalog.json`：所有 subset 列表、被跳过的文件夹（和原因）、没转换的文件、split 汇总。
- `splits/<原始 index 文件路径>.parquet`：原始数据里每个 `*_index.txt` 划分文件都转成了一个 parquet，每行是 `line`（原始行号）、`original_path`（原始写的路径）、`sample_id`（解析到 unified 里的样本，找不到就是 null）。
- `verify_report.json` / `derived_report.json`：验证结果（全部通过）。

格式细节、验证方式、读取代码见 [scripts/preprocess/README.md](../../scripts/preprocess/README.md)。

### 最小读取示例

```python
import h5py
import pyarrow.parquet as pq

root = "datasets_unified/M3N-VC/a06/individual_time_samples"
rows = pq.read_table(f"{root}/manifest.parquet").to_pylist()
with h5py.File(f"{root}/signals.h5", "r") as h5:
    audio = h5["signals/audio"][0]      # 1-D 波形，dtype 和原始文件一样
    seismic = h5["signals/seismic"][0]
print(rows[0]["labels"], rows[0]["distances_m"], rows[0]["node"])
```

### 快速查看

```bash
I="uv run python -m scripts.preprocess.inspect_unified"
$I overview                       # 每个 subset 一行
$I labels --dataset ICT           # 某个数据集的标签分布
$I subset <subset 路径>            # 元数据分布 + 信号统计
$I sample <sample_id> --original --plot s.png
$I find --label cx30 --node rs1
```

## 跨数据集的注意事项

- **dtype 不统一**：unified 版本完全保留原始 dtype（int16 / int32 / int64 / float32 / float64 都有），训练前要自己转 float 并归一化。各 subset 的 dtype 写在各自文档和 `info.json` 里。
- **有重复数据**：GracesQuarters 里很多文件夹是别的文件夹的子集；ICT 的 `ICT_*_dis` 和 M3N-VC 的 `i29` 是同一天同一批录音（采样率不同）；M3N-VC 的 `o1000` clip 和普通 clip 有一半重叠。混用之前先去重（可以用 `source_sha256` 或 run/node/clip）。
- **官方划分基本都是按 clip 随机切的**：同一个 run 里相邻的 clip 会同时出现在 train 和 test 里，时间上相邻的 clip 很像，测出来的准确率偏乐观。M3N-VC 是唯一按 run 划分 train/test 的。
- **标签拼写保留原样**：Parkland 写 `warhog`，GracesQuarters 写 `warthog`，是同一种车；只做了小写化。
- **同名车辆跨数据集**：`cx30`、`gle350`、`miata`、`mustang` 出现在 ICT 和 M3N-VC；`polaris`、`silverado`、`warthog/warhog` 出现在 Parkland 和 GracesQuarters；`mustang`、`tesla` 等出现在 Parkland。是否是同一辆车无法从数据确认。
