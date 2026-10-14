# M3N-VC

M3N-VC 是公开在 Zenodo 上的多模态、多节点车辆分类数据集（Multi-Modal Multi-Node Vehicle Classification）。它是这批数据里规模最大、元数据最全的：除了标签和距离，每个 clip 还有节点坐标、车辆坐标、精确起始时间戳，原始连续录音也一起下载了下来。

## 一眼看懂

| 项目 | 内容 |
|---|---|
| 任务 | 4 类车辆分类、背景检测（h24）、两车多标签分类（i22）、距离回归、车辆定位、跨场景泛化 |
| 车辆 | `cx30`、`gle350`、`miata`、`mustang`（和 ICT 同样 4 辆车），`background`（h24） |
| 模态 | **audio 1600 Hz（int16），seismic 200 Hz（int32）**，和其他数据集都不一样 |
| 样本长度 | 2 秒（audio 3200 点，seismic 400 点） |
| 场景 | 6 个：`a06`、`h08`、`h24`、`i22`、`i29`、`s31` |
| 节点 | 每个场景 6 或 8 个，`rs1`…`rs10` |
| 距离 | 除了 i22 和背景，几乎全有 |
| unified 样本数 | 412,564（6 个 subset，其中约一半是重叠的 `o1000` clip） |
| 原始大小 | 约 9.2 GB（clip 约 4.8 GB + Zenodo 原始文件 4.5 GB） |

## 存放位置

| 原始 `datasets/M3N-VC/...` | unified `datasets_unified/M3N-VC/...` | 样本数 | unified 大小 |
|---|---|---|---|
| `a06/individual_time_samples/` | `a06/individual_time_samples/` | 51,006 | 0.41 GB |
| `h08/individual_time_samples/` | `h08/individual_time_samples/` | 58,852 | 0.48 GB |
| `h24/individual_time_samples/` | `h24/individual_time_samples/` | 73,136 | 0.59 GB |
| `i22/individual_time_samples/` | `i22/individual_time_samples/` | 86,345 | 0.70 GB |
| `i29/individual_time_samples/` | `i29/individual_time_samples/` | 93,328 | 0.76 GB |
| `s31/individual_time_samples/` | `s31/individual_time_samples/` | 49,897 | 0.40 GB |
| `zenodo/`（原始连续录音） | **没转换** | | |

`zenodo/` 没转换的原因：它本身已经是 parquet 这种开放格式，而且上面的 clip 就是从它切出来的。在 `catalog.json` 的 `not_converted_files` 里记了 1,036 个 parquet、6 个 zip、1 个 log。

## 6 个场景

| 场景 | 日期（UTC） | 地面 | 节点 | run → 车辆（偶数 = train，奇数 = test） | 样本数 |
|---|---|---|---|---|---|
| `a06` | 2023-10-24 | 柏油路（rs2 在路沿） | rs1、2、5、6、7、8 | 0/1 miata、2/3 gle350、4/5 mustang、6/7 cx30 | 51,006 |
| `h08` | 2023-12-08 | 带碎石的柏油路 | rs1、2、5、6、7、8 | 0/1 gle350、2/3 mustang、4/5 miata、6/7 cx30 | 58,852 |
| `h24` | 2023-12-24 | 同 h08（同一个地点） | rs1、2、5、6、7、8 | 0/1 gle350、2/3 cx30、4/5 mustang、6/7 miata、**8/9 background** | 73,136 |
| `i22` | 2024-06-23 | 混凝土 | rs1、2、3、5、6、7、8、10 | **两车同时**：0/1 cx30+mustang、2/3 miata+mustang、4/5 cx30+miata | 86,345 |
| `i29` | 2024-03-29 | 混凝土（和 i22 同一片场地） | rs1、2、3、5、6、7、8、10 | 0/1 mustang、2/3 miata、4/5 gle350、6/7 cx30 | 93,328 |
| `s31` | 2023-12-31 | 带碎石的土路 | rs1、2、5、6、7、8 | 0/1 gle350、2/3 mustang、4/5 miata、6/7 cx30 | 49,897 |

- 每个场景里，同一辆车开两个 run：一个长的当 train，一个短的当 test（i29 反过来：train 15 分钟，test 约 45 分钟；i22 是 45 / 15 分钟；其他是约 30 / 10 分钟）。train/test 是 Zenodo 的 `run_ids.parquet` 里 `set` 列给的，**所有场景都是偶数 run = train、奇数 run = test**。unified 的 manifest 没有单独的 split 列，用 `run % 2` 就能分。
- **i29 和 ICT 的 `ICT_*_dis` 是同一天同一批录音**，详见 [ICT/README.md](../ICT/README.md)。不要把它们当成两个独立数据集。
- 节点数据量不均：a06 的 rs7（5,086）和 rs1（7,117）、i29 的 rs2（2,693）、s31 的 rs5（335）明显偏少，是因为部分 run 的部分节点没有录到数据。

## 文件名

```
<scene>_<车名，多车用 - 连接>_run<run>[_o1000]_<node>_<clip 五位数>[_<距离>].pt
例：a06_cx30_run6_rs1_00000_9.pt
    a06_cx30_run6_o1000_rs1_00000_10.pt
    i22_cx30-miata_run4_o1000_rs1_00000.pt        （两车，没有距离）
    h24_background_run8_o1000_rs1_00000.pt        （背景，没有距离）
```

- `_o1000`：这个 clip 的起点比普通 clip 晚 1000 ms。普通 clip 是 0–2 s、2–4 s、4–6 s…，`o1000` clip 是 1–3 s、3–5 s…，**每个 `o1000` clip 和两个普通 clip 各重叠一半**。每个场景大约一半样本是 `o1000`（unified 里 `offset_ms` 列是 0 或 1000）。只想要不重叠的数据就过滤 `offset_ms == 0`。
- 最后的距离是四舍五入到整数的米数（和 ICT、WestPoint 的向下取整不一样）。精确值在文件里。

## 原始样本里有什么

```python
{
    "data": {"rs1": {                       # key 是节点名，不是 "shake"
        "audio":   tensor[3200] int16,
        "seismic": tensor[400]  int32,
    }},
    "label": "cx30",                        # 单车是字符串；i22 是 ["cx30", "miata"]
    "distance": 9.741,                      # 米；未知（背景、i22）时是 inf
    "node_xy": [23.72, 0.36],               # 节点位置，本地坐标（米）
    "vehicle_xy": [15.69, 5.88],            # 车辆位置，本地坐标（米）；背景和 i22 是 None
    "clip_start": 1698125156.0,             # clip 起始时间，unix 秒
    "scene": "a06",
    "run": 6,
    "clip": 0,
}
```

和其他数据集不同：没有 `flag` 字段，数组本身就是一维的（不用压平）。

unified 的 manifest 把这些都拆成了列：`labels`、`distances_m`、`node`、`run`、`clip`、`scene`、`offset_ms`、`node_xy`、`vehicle_xy`、`clip_start`、`name_labels`。

## 距离覆盖

| 场景 | 有距离 | 范围 |
|---|---|---|
| a06 | 51,006 / 51,006 | 0.6–96 m |
| h08 | 58,763 / 58,852 | 0.06–66 m |
| h24 | 58,908 / 73,136（背景 14,228 个没有） | 0.2–67 m |
| i22 | 0 / 86,345（两车，原始就是 inf） | |
| i29 | 93,067 / 93,328 | 0.6–192 m |
| s31 | 49,868 / 49,897 | 0.1–186 m |

h08/h24 的场地比较小，最远不到 70 m；i29/s31 的车会开到将近 200 m 外。

## Zenodo 原始文件：`datasets/M3N-VC/zenodo/`

```
zenodo/
  download.log                    内容就是 "DONE"
  <scene>.zip                     从 Zenodo 下载的压缩包（6 个，解压后就是下面的文件夹）
  <scene>/
    run_ids.parquet               每个 run 的 label、set（train/test）、开始/结束时间、时长
    sensor_location.parquet       每个节点的经纬度和地面类型（terrain）
    run<k>_gps.parquet            车辆 GPS 轨迹：time、latitude、longitude、elevation（每秒一行）
    run<k>_<node>_dis.parquet     车到这个节点的距离：time、distance（每秒一行）
    run<k>_<node>_geo.parquet     geophone 连续录音：timestamp、channel（EH3）、samples，一行一个采样点
    run<k>_<node>_mic.parquet     麦克风连续录音：timestamp、samples，一行一个采样点
```

- i22 没有 `gps` 文件（两车同时开，没有单一车辆轨迹），所以 i22 的 clip 没有 `vehicle_xy` 和距离。
- a06 有 7 个、s31 有 9 个 `mic.parquet` 是空的（0 行），对应节点的 clip 数会少。
- 原始 geophone 实际采样率约 197.9 Hz（不是严格 200 Hz），切 clip 时被插值成了 200 Hz，见下面的「a06 地震数据缺口」。
- 想做比 2 秒更长的时间窗口、或者自己重新切 clip，就从这里读。

## 注意事项

- **采样率和 dtype 跟其他数据集都不一样**（1600/200 Hz，int16/int32）。和其他数据集一起训练时，要先重采样到统一的采样率。
- **`o1000` 重叠**：随机划分时，一个普通 clip 和覆盖它一半的 `o1000` clip 很可能分到两边。官方是按 run 分 train/test 的，按 run 分就没这个问题。
- **a06 地震数据缺口**：大约 8% 的 a06 clip 里有 0.1 秒以上的 seismic 片段是完全线性或者完全不变的，也就是原始录音丢了数据、被 M3N-VC 的预处理插值或者补成常数。我抽查了其他场景各 500 个 clip，没发现这种情况。用 a06 时可以考虑把这些 clip 去掉或者 mask 掉。
- **i22 是多标签**：`labels` 有两个元素，没有距离和车辆位置。h24 的 background 也没有距离。
- **和 ICT 重复**：i29 = ICT 2024-03-29。
- `node_xy` / `vehicle_xy` 是每个场景自己的本地平面坐标（米），不同场景之间不能直接比较。
