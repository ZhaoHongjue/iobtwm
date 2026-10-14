# ICT

ICT 这个名字下有三个文件夹，分两类：

1. `ICT_train_dis/`、`ICT_val_dis/`：2024-03-29 采的真实录音，4 辆民用车，每个 clip 带车-传感器距离（`dis` = distance）。
2. `ict_pt_files/`：把两辆车的信号叠加成一个样本的**合成**数据，4 秒长。

ICT 应该是 Illinois Center for Transportation（伊利诺伊州 Rantoul）的试验场：M3N-VC 的 `i22`、`i29` 场景的 GPS 坐标也在那里。

## 一眼看懂

| 项目 | 内容 |
|---|---|
| 任务 | 4 类车辆分类、距离回归（`*_dis`）、两车多标签分类（`ict_pt_files`） |
| 车辆 | `cx30`（Mazda CX-30）、`gle350`（Mercedes GLE 350）、`miata`（Mazda MX-5 Miata）、`mustang`（Ford Mustang） |
| 模态 | audio 8000 Hz，seismic 100 Hz |
| 样本长度 | `*_dis`：2 秒（16000 + 200 点）；`ict_pt_files`：4 秒（32000 + 400 点） |
| 节点 | `rs1`、`rs2`、`rs3`、`rs5`、`rs6`、`rs7`、`rs8`、`rs10`（8 个） |
| 距离 | `*_dis` 每个 clip 都有 |
| unified 样本数 | 61,870（3 个 subset） |
| 原始大小 | 约 10.8 GB |

## 文件夹一览

| 原始 `datasets/...` | unified `datasets_unified/...` | 样本数 | dtype | unified 大小 |
|---|---|---|---|---|
| `ICT_train_dis/2024-03-29-ICT/individual_time_samples/` | 同路径 | 11,615 | int64 | 1.51 GB |
| `ICT_val_dis/2024-03-29-ICT/individual_time_samples/` | 同路径 | 38,255 | int64 | 4.96 GB |
| `ict_pt_files/` | `ict_pt_files/` | 12,000 | float32 | 1.56 GB |

在 unified 的 manifest 里这三个 subset 的 `dataset` 列都是 `ICT`。

## `ICT_train_dis` / `ICT_val_dis`：真实录音

### 采集方式

每个 run 只有一辆车在场地里开，8 个节点同时录。train 和 val 按 run 分开，**不会有同一个 run 的 clip 跨两边**：

| 文件夹 | run → 车辆（clip 数，8 个节点合计） | 距离范围 |
|---|---|---|
| `ICT_train_dis` | 0 mustang 3,098；2 miata 2,608；4 gle350 2,905；6 cx30 3,004 | 1.7–191.6 m |
| `ICT_val_dis` | 1 mustang 9,226；3 miata 10,063；5 gle350 9,211；7 cx30 9,755 | 0.6–188.5 m |

val 比 train 大三倍多，因为 val 的 run 更长（大约 45 分钟 vs 15 分钟）。

### 和 M3N-VC `i29` 的关系

M3N-VC 的 `i29` 场景也是 2024-03-29，同样 4 辆车、同样 8 个节点，run 0/2/4/6 是 train（各 900 秒）、run 1/3/5/7 是 test，车辆顺序完全一样。所以 `ICT_*_dis` 和 `M3N-VC/i29` 几乎可以肯定是**同一批录音的两种处理**：

| | ICT `*_dis` | M3N-VC `i29` |
|---|---|---|
| 采样率 | 8000 / 100 Hz | 1600 / 200 Hz |
| dtype | int64 | int16 / int32 |
| 每个 run 的 clip 数 | 略多（例如 run 0：3,098） | 略少（run 0：2,900，不算 o1000） |
| 额外信息 | 只有距离 | 距离、坐标、时间戳 |

两边的 clip 不是一一对应的（切法不同），所以没法直接按行对齐。**不要把两者当成两个独立数据集**，例如别用一个训练、另一个当「跨数据集」测试。

### 文件名

```
<车名>_run<run>_<node>_<clip>_<距离>.pt
例：cx30_run6_rs1_0_187.pt
```

最后的数字是距离的整数部分（向下取整，单位米），例如 `187` 对应 187.14 m。精确值在文件里。

### 原始样本里有什么

```python
{
    "flag": {"shake": {"audio": True, "seismic": True}},
    "label": "cx30",                       # 字符串
    "distance": np.float64(187.138...),    # 米
    "data": {"shake": {
        "seismic": ndarray[10, 20]   int64,  # 10 个 0.2 s 区间
        "audio":   ndarray[10, 1600] int64,
    }},
}
```

unified 里波形压平成 `[200]` / `[16000]`，`labels = ["cx30"]`、`distances_m = [187.138...]`，还有 `node`、`run`、`clip`、`name_labels`（文件名里的车名）。

## `ict_pt_files`：合成两车混合

- 12 种**有序**车辆对（4 × 3），每种 1,000 个，共 12,000 个：`cx30_gle350`、`cx30_miata`、`cx30_mustang`、`gle350_cx30`、…、`mustang_miata`。`a_b` 和 `b_a` 是分开的两组。
- 文件名：`<车A>_<车B>_mixed_<i>.pt`，`i` 从 0 到 999，只是编号，没有时间顺序。
- 没有节点、run、距离信息。
- 原始样本：

  ```python
  {
      "label": tensor([3, 1]),                       # int64，两辆车的 id
      "data": {"shake": {
          "audio":   tensor[32000] float32,
          "seismic": tensor[400]   float32,
      }},
      "flag": {"shake": {"audio": True, "seismic": True}},
  }
  ```

- 标签 id 的对应关系是从文件名反推出来的（`specs.ICT_MIXED_LABELS`），和所有文件名都一致：0 = miata、1 = gle350、2 = mustang、3 = cx30。unified 的 `labels` 是 `["cx30", "gle350"]` 这样的名字列表，顺序和原始 id 顺序一致。
- **采样率是推断的**：原始数据没有记录。32000 / 400 点和其他 ICT 数据一样是 80 : 1，所以按 8000 / 100 Hz、4 秒处理（写在 `info.json` 的 `note` 里）。如果以后找到原始脚本，要再核实。
- 是由哪些单车 clip 合成的、怎么合成的（直接相加还是加权），数据里看不出来。

## 官方划分

ICT 没有附带任何 `*_index.txt`。文件夹名本身就是划分：`ICT_train_dis` = train，`ICT_val_dis` = val，没有 test。cs537aiot 配置里提到的 `indices/single/ICTMixed/...` 不在本地。

如果要自己给 `ict_pt_files` 分 train/test，注意它是合成数据，不同样本可能复用了同一段原始 clip。

## 注意事项

- **和 M3N-VC i29 重复**，见上。
- dtype 是 int64（真实录音）和 float32（合成），数值量纲很可能不同，混用前分别归一化。
- 节点 `rs2` 的数据明显少（train 895、val 2,443，其他节点 1,500–5,600），`rs7` 在 train 里也偏少。
- 车辆类别基本平衡。
