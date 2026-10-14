# WestPoint

`2026-04-14-West-Point` 是最新的一批数据：2026-04-14 在 West Point 采集，两辆车轮流在场地里开，8 个节点同时录 audio 和 seismic，每个 clip 都带车-传感器距离。数据干净、整齐，没有合成数据，也没有重复文件夹。

## 一眼看懂

| 项目 | 内容 |
|---|---|
| 任务 | 2 类车辆分类、距离回归、少样本训练（官方有 5 分钟训练集） |
| 车辆 | `kona`（Hyundai Kona）、`wagoneer`（Jeep Wagoneer） |
| 模态 | audio 8000 Hz，seismic 100 Hz |
| 样本长度 | 2 秒（audio 16000 点，seismic 200 点） |
| 节点 | `rs1`、`rs2`、`rs3`、`rs5`、`rs6`、`rs7`、`rs8`、`rs10`（8 个，每个约 3,600 个 clip） |
| 距离 | 28,649 / 28,804 有，范围 0.7–273.9 m |
| unified 样本数 | 28,804（1 个 subset） |
| 原始大小 | 约 1.9 GB |

## 存放位置

| 原始 `datasets/2026-04-14-West-Point/...` | unified `datasets_unified/...` | 内容 |
|---|---|---|
| `individual_time_samples/` | `2026-04-14-West-Point/individual_time_samples/`（1.87 GB） | 28,804 个样本，float32 |
| `indices/` | `splits/2026-04-14-West-Point/indices/{train,val}_index.parquet` | 完整划分 |
| `indices_5minutes/` | `splits/2026-04-14-West-Point/indices_5minutes/...` | 每个 run 只用前 5 分钟训练 |
| `indices_filtered/` | `splits/2026-04-14-West-Point/indices_filtered/...` | 5 分钟训练集再筛一遍 |

unified 里这个 subset 的 `dataset` 列是 `WestPoint`。

## 采集方式

11 个 run，每个 run 只有一辆车：

| run | 车辆 | clip 数（8 个节点合计） | 每个节点大约时长 | 官方划分 |
|---|---|---|---|---|
| 0、1、2 | kona | 3,926、3,799、3,638 | 15–16 分钟 | train |
| 3、4、5 | kona | 1,272、1,238、1,352 | 5 分钟 | val |
| 6、7、8 | wagoneer | 3,712、3,632、3,992 | 15–16 分钟 | train |
| 9、10 | wagoneer | 1,278、965 | 4–5 分钟 | val |

- 类别：kona 15,225、wagoneer 13,579，基本平衡。
- 同一个 run 里，`clip` 编号就是时间顺序（每个 clip 2 秒），同一个 `(run, clip)` 在 8 个节点上是同一时刻的 8 个视角。

## 文件名

```
run<run>_<node>_<clip>__<距离>.pt
例：run0_rs1_0__180.pt、run5_rs5_22__35.pt
```

- 注意距离前面是**两个下划线**。
- 距离是向下取整的米数（例如 `180` 对应 180.71 m）；未知时写 `-1`。
- 文件名里没有车名，车名只在文件内容里。

## 原始样本里有什么

```python
{
    "flag": {"shake": {"audio": True, "seismic": True}},
    "label": {"kona"},                          # Python set，只有一个元素
    "distance": {"kona": np.float64(180.71)},   # {车名: 米}
    "coordinates": {},                          # 全部为空
    "data": {"shake": {
        "seismic": tensor[1, 10, 20]   float32,  # 10 个 0.2 s 区间
        "audio":   tensor[1, 10, 1600] float32,
    }},
}
```

- `label` 是 set，说明格式上预留了多车的情况，但这批数据每个样本都只有一辆车。
- `coordinates` 在所有 28,804 个样本里都是空 dict，没有位置信息。
- **155 个 clip 没有距离**：原始值是 NaN（文件名写 `-1`），集中在 run 8（66 个）、run 1（56 个）、run 4（25 个）、run 10（8 个）。unified 的 `distances_m` 为 `[null]`。

unified 里波形压平成 `[16000]` / `[200]`，可用的列有 `labels`、`distances_m`、`node`、`run`、`clip`。

## 官方划分

三套划分的 **val 完全一样**（run 3、4、5、9、10 全部，6,105 个），区别只在 train。没有 test。train 和 val 按 run 分开，不会有同一个 run 的 clip 跨两边。

| 划分 | train | 怎么来的 |
|---|---|---|
| `indices/` | 22,699 | run 0、1、2、6、7、8 的全部 clip |
| `indices_5minutes/` | 7,200 | 同样 6 个 run，每个节点只取 clip 0–149（前 150 个 = 前 5 分钟），6 × 8 × 150 = 7,200 |
| `indices_filtered/` | 5,040 | 从 `indices_5minutes` 的 train 里再挑约 70%，每个 run 约 840 个 |

`indices_filtered` 的筛选规则数据里看不出来：我检查过，不是按距离筛的（留下和扔掉的距离分布差不多），各节点被扔掉的比例也一样。如果要用，最好去问数据提供者。

划分文件里的原始路径是 `/home/tkimura4/data/datasets/2026-04-14-West-Point/...`，6 个文件共 53,254 行全部解析到了本地样本。

## 注意事项

- 只有两类，适合做二分类或者当作其他数据集的跨域测试，单独用来做分类太简单。
- 距离最远 274 m，远距离的 clip 里车的信号可能很弱，做分类时可以考虑按距离过滤。
- 节点之间的位置关系没有记录（`coordinates` 为空），没法像 M3N-VC 那样用节点坐标。
