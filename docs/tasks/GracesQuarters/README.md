# GracesQuarters

GracesQuarters（GQ）是在野外试验场 Graces Quarters 采的车辆数据。车辆主要是越野车和无人地面车：`polaris`（全地形车）、`warthog`、`husky`（无人车，有时有人跟着走）、`silverado` / `truck`（皮卡）。传感器节点叫 `gq-1` ~ `gq-4`。

这个文件夹是所有数据集里最乱的：有两代数据（老的 `day_*` 和 2024 年 7–8 月的新采集），还有很多从老数据里挑出来的子集和合成混合数据。看完下面的「文件夹关系」再用。

## 一眼看懂

| 项目 | 内容 |
|---|---|
| 任务 | 单车分类（4 类）、背景检测（有车/没车）、多车多标签分类、距离回归（2024-08-06/07）、合成混合分离 |
| 模态 | audio 8000 Hz，seismic 100 Hz |
| 样本长度 | 2 秒（audio 16000 点，seismic 200 点） |
| 节点 | `gq-1`、`gq-2`、`gq-3`（老数据）；2024 数据多一个 `gq-4` |
| 距离 | 只有 `2024-08-06-GQ`、`2024-08-07-GQ(_past)` 有 |
| unified 样本数 | 143,987（18 个 subset，其中大量重复） |
| 原始大小 | 约 22 GB |

## 文件夹关系（先看这个）

```
老数据（两天，day 1 = run 0–6，day 2 = run 12–30）
  day_1_sensor_all  ┐
  day_2_sensor_all  ┘ ← 全集，其他老数据都是从这里挑出来的
    ├─ GracesQuartersFiltered            挑出「有车」的 clip（两天合在一起）
    ├─ GracesQuartersFilteredNone        挑出「没车」的 clip，标签改成 background
    ├─ GracesQuartersDay{1,2}Filtered            同上，按天分开（又单独挑了一次，和上面不完全一样）
    ├─ GracesQuartersDay{1,2}FilteredNone
    ├─ GracesQuartersDay{1,2}Filtered_random     数量相同但挑法不同（随机）的对照组
    └─ GracesQuartersDay{1,2}FilteredNone_random
  GracesQuartersMixed                    两辆车的 clip 合成叠加（合成数据）

2024 新采集
  2024-07-19-GQ-Background               纯背景
  2024-08-05-GQ                          背景 + truck + polaris，单车
  2024-08-06-GQ                          单车和两车同时，有距离
  2024-08-07-GQ                          单车、两车、三车、背景，有距离
    └─ 2024-08-07-GQ_past                上面去掉背景 run 16 之后的旧版本，完全包含在里面
```

我逐个核对过：

- 所有 `*Filtered` / `*Filtered_random` 文件夹里的文件，和 `day_*_sensor_all` 里同名文件**字节完全相同**。
- 所有 `*FilteredNone*` 文件夹里的 audio/seismic 和 `day_*_sensor_all` 里同名文件**完全相同**，只是 label 被改成了字符串 `"background"`（所以文件字节不同）。也就是说，同一段信号在 `sensor_all` 里标的是车名，在 `FilteredNone` 里标的是 background。它的意思应该是「这次 run 里车离得远、没被检测到的片段」。
- `Filtered` 和 `FilteredNone` 文件名没有交集。
- `GracesQuartersFiltered` ≠ `Day1Filtered` ∪ `Day2Filtered`，大约 2,000 个文件不同，是分别挑的。
- `Day1Filtered_random` 和 `Day1Filtered` 有 4,013 / 4,738 个重叠。
- `2024-08-07-GQ_past` 的每个文件都在 `2024-08-07-GQ` 里（字节相同），`2024-08-07-GQ` 只多了 run 16 的 1,609 个背景 clip。

**训练时别把这些文件夹简单拼在一起**，重复会非常严重。一般用法是：老数据只用 `day_1_sensor_all` + `day_2_sensor_all`（或官方划分），2024 数据用 `2024-08-07-GQ` 而不是 `_past`。

## 文件夹一览

所有 unified 路径都是 `datasets_unified/GracesQuarters/` + 同样的相对路径。

| 原始文件夹 `datasets/GracesQuarters/...` | 样本数 | 标签 | dtype | unified 大小 |
|---|---|---|---|---|
| `day_1_sensor_all/` | 15,796 | silverado 5,034、polaris 4,792、husky 3,732、warthog 2,238 | int64 | 2.05 GB |
| `day_2_sensor_all/` | 19,366 | polaris 7,207、warthog 5,712、silverado 3,225、husky 3,222 | int64 | 2.51 GB |
| `GracesQuartersFiltered/` | 10,547 | silverado 4,368、polaris 3,250、warthog 1,721、husky 1,208 | int64 | 1.37 GB |
| `GracesQuartersFilteredNone/` | 10,547 | background | int64 | 1.37 GB |
| `GracesQuartersDay1Filtered/` | 4,737 | silverado 1,510、polaris 1,437、husky 1,119、warthog 671 | int64 | 0.61 GB |
| `GracesQuartersDay1FilteredNone/` | 4,737 | background | int64 | 0.61 GB |
| `GracesQuartersDay1Filtered_random/` | 4,738 | silverado 2,083、polaris 1,589、husky 653、warthog 413 | int64 | 0.61 GB |
| `GracesQuartersDay1FilteredNone_random/` | 4,738 | background | int64 | 0.61 GB |
| `GracesQuartersDay2Filtered/` | 5,808 | polaris 2,162、warthog 1,713、silverado 967、husky 966 | int64 | 0.75 GB |
| `GracesQuartersDay2FilteredNone/` | 5,808 | background | int64 | 0.75 GB |
| `GracesQuartersDay2Filtered_random/` | 5,809 | silverado 2,285、polaris 1,661、warthog 1,308、husky 555 | int64 | 0.75 GB |
| `GracesQuartersDay2FilteredNone_random/` | 5,809 | background | int64 | 0.75 GB |
| `GracesQuartersMixed/` | 6,000 | 6 种两车组合各 1,000 | float64 | 0.78 GB |
| `2024-07-19-GQ-Background/individual_time_samples/` | 3,562 | background | float32 | 0.23 GB |
| `2024-08-05-GQ/individual_time_samples/` | 5,203 | background 3,237、truck 983、polaris 981、polairs 2 | float32 | 0.34 GB |
| `2024-08-06-GQ/individual_time_samples/` | 7,155 | 见下 | float32 | 0.46 GB |
| `2024-08-07-GQ/individual_time_samples/` | 12,618 | 见下 | float32 | 0.82 GB |
| `2024-08-07-GQ_past/individual_time_samples/` | 11,009 | 同上去掉 background | float32 | 0.71 GB |
| `indices/{GQ,GQ_Parking,Parking_Lot}/` | | 划分文件 | | `splits/GracesQuarters/indices/...` |

## 老数据：`day_1_sensor_all` / `day_2_sensor_all`

- 每个 run 只有一辆车，3 个节点同时录：

  | 天 | run → 车辆（clip 数，3 个节点合计） |
  |---|---|
  | day 1 | 0 polaris 2,713；1 polaris 2,079；2 silverado 5,034；3/4/5 husky 1,224/1,074/1,434；6 warthog 2,238 |
  | day 2 | 12 silverado 791；13 warthog 1,462；14 husky 1,614；16 polaris 596；19 polaris 2,873；23 polaris 533；24 warthog 1,075；25 husky 1,608；26 silverado 709；27 warthog 1,432；28 polaris 3,205；29 warthog 1,743；30 silverado 1,725 |

- 标签是 int64 标量 id，对应关系是从文件名反推出来的（`specs.GQ_LABELS`）：0 = polaris、1 = warthog、2 = silverado、3 = husky。
- 每个样本还多两个字段，unified 只放在 `original_meta` 里、没进 `labels`：
  - `targets`：场上实际有什么，例如 `["husky", "pedestrian"]`（husky 和 warthog 的 run 都有人跟着走）、`["polaris", "fog-machine"]`（day 2 run 19 开了烟雾机）、`["sedan"]`（day 1 的 silverado run 写的是 sedan）。
  - `type`：全部是 `"single_car"`。
- 原始 dtype 是 int64（波形本身是整数 ADC 读数）。

文件名：`<车名>_run<run>_gq_<node>_<clip>.pt`，例如 `husky_run3_gq_1_0.pt`。注意节点写成 `gq_1`，unified 的 `node` 列统一成 `gq-1`。

## 合成数据：`GracesQuartersMixed`

- 6 种有序组合：`polaris_silverado`、`polaris_warthog`、`silverado_polaris`、`silverado_warthog`、`warthog_polaris`、`warthog_silverado`，每种 1,000 个，文件名 `<车A>_<车B>_mixed_<i>.pt`。
- 看起来是把两辆车各自的单车 clip 叠加成一个「两车同时经过」的样本，用来做多标签分类或者分离。没有节点、run 信息。
- 原始数组形状 `[16000, 1]` / `[200, 1]`，float64。
- **标签坑**：原始存的整数标签是错的，6 种组合里有 5 种都存成 `(0, 1)`。unified 的 `labels` 从文件名取（例如 `["polaris", "silverado"]`），原始值在 `original_meta`。

## 2024 新采集

四天的数据，节点 `gq-1` ~ `gq-4`，float32，原始形状 `[1, 10, 1600]` / `[1, 10, 20]`（10 个 0.2 秒区间）。

### `2024-07-19-GQ-Background`、`2024-08-05-GQ`

- 文件名带车名和距离：`<车名>_run<run>_<node>_<clip>_<距离>.pt`，例如 `truck_run1_gq-1_351_-1.pt`。这两天距离全是 `-1`（未知），原始 `distance` 字段是 `inf`，unified 的 `distances_m` 为 null。
- 07-19 全是背景（run 0），gq-4 只有 99 个 clip。
- 08-05：run 0 背景、run 1 truck、run 2 polaris。
- 原始 `label` 大多是字符串，08-05 里有 7 个是长度为 1 的 object 数组。unified 的 `labels` 已经统一成字符串列表。

### `2024-08-06-GQ`

- 文件名不带车名：`run<run>_<node>_<clip>.pt`，标签和距离都在文件里。
- 66 个短 run（每个 run 每节点几十个 clip），标签是 object 数组，**多车时有多个元素**：

  | 标签 | clip 数 | run |
  |---|---|---|
  | `warthog` | 3,496 | 0–26（缺 12、13）、67–70 |
  | `polaris` | 545 | 27–34（缺 30） |
  | `warthog+warthog` | 556 | 37–41（两辆 warthog） |
  | `truck` | 967 | 42–52 |
  | `polaris+truck` | 541 | 53–56 |
  | `warthog+polaris` | 1,050 | 57–66 |

- `distance` 是 `{车名: 米}` 的 dict，所有 clip 都有，范围 0.8–78 m。unified 的 `distances_m` 和 `labels` 一一对齐。两辆 warthog 的 run 因为 dict 的 key 相同，原始只存了一个距离，unified 里两个 warthog 的 `distances_m` 是同一个数，分不出哪辆是哪辆。

### `2024-08-07-GQ`（和 `_past`）

| run | 标签 | clip 数 |
|---|---|---|
| 0、2、6 | `polaris` | 260、2,380、878 |
| 3 | `warthog+truck` | 1,932 |
| 4 | `warthog+polaris` | 1,392 |
| 5 | `truck` | 1,937 |
| 7、10、12 | `warthog+polaris+truck` | 723、412、442 |
| 8、9 | `warthog` | 405、248 |
| 16 | `background` | 1,609（只在 `2024-08-07-GQ` 里有） |

- 车辆 clip 全都有距离，范围 0–135 m；背景 clip 没有距离。
- 有三车同时出现的数据，是做多标签/计数的好材料。

## 官方划分

| 划分 | train | val | test | 指向 |
|---|---|---|---|---|
| `indices/GQ/` | 28,126 | 3,515 | 3,521 | `day_1_sensor_all` + `day_2_sensor_all`，合起来正好是两者全部 35,162 个 |
| `indices/GQ_Parking/` | 29,289 | 3,646 | 3,652 | 上面的 GQ 全部 + `Parking_Lot` 的样本 |
| `indices/Parking_Lot/` | 1,163 | 131 | 131 | `2023-06-18-UIUC-Parking-Lot-B4`，**本地没有**，`sample_id` 全是 null |

- 划分是**逐 clip 随机**的：train 和 test 都包含全部 20 个 run，相邻 clip 会跨集合，分数偏乐观。
- val 和 test 不重叠（和 Parkland 不同）。
- 2024 数据没有附带划分文件。cs537aiot 配置里提到的 `indices/single/2024-08-06-GQ/...` 等文件不在本地。

## 注意事项

- **重复数据**，见上面的「文件夹关系」。
- **`FilteredNone` 的 background 标签和 `sensor_all` 冲突**：同一段信号两边标签不一样。unified 的 `name_labels`（从文件名取）是车名、`labels` 是 background，`info.json` 里 `name_label_mismatches` 等于样本总数，这是预期的。
- **拼写**：`warthog`（这里）和 Parkland 的 `warhog` 是同一种车；老数据叫 `silverado`，2024 数据叫 `truck`，是否是同一辆车不确定；`2024-08-05-GQ` 有 2 个 clip 的标签是错别字 `polairs`，unified 保留原样。
- **dtype**：老数据 int64，Mixed float64，2024 数据 float32，量纲可能也不同，混用前先各自归一化。
- `2024-07-19` 的 `gq-4` 只有 99 个 clip，节点很不平衡。
