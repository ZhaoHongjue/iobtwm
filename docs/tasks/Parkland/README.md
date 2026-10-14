# Parkland

Parkland 是这批数据里最老的一份，也是 FoundationSense / cs537aiot 项目一直在用的车辆数据（他们代码里的路径叫 `MOD/Parkland`）。场景是车辆（以及步行的人）从几个传感器节点旁边经过，每个节点同时录 audio 和 seismic。

## 一眼看懂

| 项目 | 内容 |
|---|---|
| 任务 | 单目标分类（11 类：车辆 + 步行），自监督预训练，带风噪的鲁棒性测试 |
| 模态 | audio 8000 Hz（原始 16 kHz 降采样 2 倍），seismic 100 Hz |
| 样本长度 | 2 秒（audio 16000 点，seismic 200 点），不重叠切分 |
| 节点 | `rs1`、`rs2`、`rs3`、`rs7` |
| 距离 | 无 |
| unified 样本数 | 78,521（4 个 subset） |
| 原始大小 | 约 25 GB |

## 类别

标签编号来自 `cs537aiot/src/data_preprocess/Parkland/extract_samples.py`（unified 版本的 `specs.PARKLAND_LABELS`）：

| id | 名字 | 是什么 |
|---|---|---|
| 0 | `polaris` | Polaris 全地形车 |
| 1 | `warhog` | Warthog 无人车（原始拼写就是 `warhog`，GracesQuarters 写 `warthog`） |
| 2 | `silverado` | Chevrolet Silverado 皮卡 |
| 3 | `motor` | 摩托车 |
| 4 | `tesla` | Tesla 轿车 |
| 5 | `mustang` | Ford Mustang |
| 6 | `walk` | 人走路 |
| 7 | `bicycle` | 自行车 |
| 8 | `forester` | Subaru Forester |
| 9 | `pickup` | 皮卡 |
| 10 | `scooter` | 电动滑板车 |

cs537aiot 的标准分类任务只用前 7 类（0–6）；7–10 只出现在 `extra_time_samples` 和 `individual_samples` 里，主要用来做预训练。

## 文件夹一览

| 原始文件夹 `datasets/Parkland/...` | unified `datasets_unified/Parkland/...` | 样本数 | 内容 |
|---|---|---|---|
| `individual_time_samples/` | `individual_time_samples/`（14.2 GB） | 14,670 | 7 类主数据集，时域。7,335 个普通文件 + 7,335 个 `_dwt` 文件 |
| `individual_time_samples_with_wind/` | `individual_time_samples_with_wind/`（0.48 GB） | 7,335 | 同一批 clip，audio 加了风噪 |
| `extra_time_samples/` | `extra_time_samples/`（2.1 GB） | 32,274 | 额外录音（11 类里的 8 类），用来预训练 |
| `individual_samples/` | `individual_samples/`（3.1 GB） | 24,242 | 老版本数据，**频域**（FFT），约 17k 个样本只存在这里 |
| `individual_freq_samples/` | 没转换 | 7,335 | `individual_time_samples` 的 FFT，可重新算 |
| `individual_freq_samples_with_wind/` | 没转换 | 7,335 | `..._with_wind` 的 FFT |
| `extra_freq_samples/` | 没转换 | 32,274 | `extra_time_samples` 的 FFT |
| `time_data_partition/` | `splits/Parkland/time_data_partition/*.parquet` | | 时域数据的 train/val/test/pretrain 划分 |
| `time_data_partition_with_wind/` | `splits/Parkland/time_data_partition_with_wind/*.parquet` | | 风噪版本的划分 |
| `data_partition/` | `splits/Parkland/data_partition/*.parquet` | | 老频域数据的划分 |

三个 `*_freq_*` 文件夹没转换，是因为它们每个文件都是同名时域文件逐 0.2 秒做 FFT 的结果。`scripts/preprocess/check_derived.py` 逐个文件验证过：标签、flag 完全一样，频谱和用 unified 波形重算的 FFT 一致（最大相对误差约 6e-8，只是 float32 舍入）。要用频域就自己从时域算。

## 各 subset 详解

### `individual_time_samples`：主数据集

- 12 段录音（recording），录音名就写在文件名里，表示哪辆车、哪个时间：

  | recording | 类别 | 普通 clip 数 |
  |---|---|---|
  | `Polaris0150pm`、`Polaris0215pm`、`Polaris0235pm-NoLineOfSight` | polaris | 520 + 424 + 447 |
  | `Warhog1135am`、`Warhog1149am`、`Warhog-NoLineOfSight` | warhog | 276 + 432 + 95 |
  | `Silverado0255pm`、`Silverado0315pm` | silverado | 521 + 489 |
  | `motor` | motor | 903 |
  | `tesla` | tesla | 1,034 |
  | `mustang0528` | mustang | 1,152 |
  | `walk2` | walk | 1,042 |

  `NoLineOfSight` 表示车和传感器之间有遮挡，看不到车。
- 只有 `rs1`、`rs3` 两个节点。
- **每个 clip 有两个文件**：`<recording>_<node>_<clip>.pt` 和 `<recording>_<node>_<clip>_dwt.pt`。我核对过，`_dwt` 文件里的 audio/seismic 波形和普通文件**逐字节相同**，它多出来的是两种小波（`db4`、`sym8`）分解系数。在 unified 里：
  - 两个文件各占 `signals/audio`、`signals/seismic` 的一行（所以 N = 14,670 = 7,335 × 2）。
  - 小波系数放在 `extras/wavelet/{db4,sym8}/shake/{audio,seismic}`，每个 7,335 行，形状 audio `[1, 1, 14, 16000]`、seismic `[1, 1, 8, 200]`（14/8 个尺度，每个尺度和原信号等长）。
  - 这 14 GB 几乎都是小波系数。只要波形的话，过滤掉 `sample_id` 以 `_dwt` 结尾的行就行。
- dtype：float32。

### `individual_time_samples_with_wind`：加风噪版本

- 和上面的普通文件一一对应，文件名相同（没有 `_dwt`）。
- 我核对过：**seismic 和干净版本完全一样，audio 不一样**，而且能量大很多（标准差大 2–9 倍）。cs537aiot 里有 `wind_noise_generator.py`（Nelke & Vary 2014 风噪模型），应该就是用它合成风噪加到 audio 上。
- 用途：测模型对风噪的鲁棒性。

### `extra_time_samples`：预训练用的额外数据

- 14 段录音：`bicycle`、`bicycle2`、`forester`、`forester2`、`motor`、`motor2`、`mustang0528`、`pickup`、`pickup2`、`scooter`、`scooter2`、`tesla`、`walk`、`walk2`。
- 8 个类别：scooter 5,694、pickup 4,959、motor 4,926、forester 4,388、bicycle 4,238、tesla 3,410、walk 3,106、mustang 1,553。
- 4 个节点：`rs1`、`rs2`、`rs3`、`rs7`。
- dtype：float32。
- 在官方划分里整个文件夹都放进 `pretrain_index`（无标签预训练），没有出现在 train/test 里。

### `individual_samples`：老的频域版本

- 唯一一个 `representation = "fft"` 的 subset。每个样本是 10 个 0.2 秒区间分别做 FFT 的结果：audio `[2, 10, 1600]`、seismic `[2, 10, 20]`，第一维 0 = 实部、1 = 虚部。unified 保留这个原始形状，不压平。
- 15 段录音，4 个节点（`rs1`、`rs2`、`rs3`、`rs7`），9 类。
- 为什么要保留它：约 17k 个样本（`mustang`、`tesla`、`pickup`、`forester`、`motor` 这几段老录音）在任何时域文件夹里都找不到。
- **标签坑**：其中 16,907 个老样本存的整数 id 用的是另一套编号（例如 forester 和 mustang 都存成 3），和上面的表对不上。unified 的 `labels` 一律从文件名里的录音名推出来，原始 id 保留在 `original_meta` 里。
- 官方划分 `data_partition/` 引用的就是这个文件夹（原路径 `/home/dongxin/data/Parkland/individual_samples/...`）。

## 文件名规则

```
<recording>_<node>_<clip>[_dwt].pt
例：Polaris0150pm_rs1_0.pt、walk2_rs3_1041_dwt.pt、mustang0528_rs3_115.pt
```

- `recording`：录音名，开头是类别名（大小写不一），后面可能带时间（`0150pm`）、日期（`0528`）、序号（`2`）或 `-NoLineOfSight`。
- `node`：传感器节点。
- `clip`：这段录音里的第几个 2 秒片段，从 0 开始，相邻编号时间上相邻。

## 原始样本里有什么

```python
{
    "label": tensor(0),                                   # int64 标量，类别 id
    "flag": {"shake": {"audio": True, "seismic": True}},  # 这个模态是否有效
    "data": {"shake": {
        "audio":   tensor[1, 10, 1600] float32,           # 10 个 0.2 s 区间 × 1600 点
        "seismic": tensor[1, 10, 20]   float32,
    }},
    # 只有 _dwt 文件有：
    "wavelet": {"db4": {...}, "sym8": {...}},
}
```

`_dwt` 文件里的 `data` 形状是 `[1, 1, 16000]` / `[1, 1, 200]`，数值和普通文件一样。

unified 版本里波形被压平成 `[16000]` / `[200]`（按时间顺序的纯 reshape），原始形状记在 `original_meta`。manifest 里对 Parkland 有用的列：`labels`、`node`、`clip`、`recording`、`name_labels`。`run`、`distances_m` 都是空的。

## 官方划分

划分文件里的路径指向别的机器（`/data/tkimura4/...`、`/home/dongxin/...`、`/home/sl29/...`），unified 按（顶层文件夹, 父文件夹, 文件名）解析到本地样本。

| 划分 | train | val | test | 指向 |
|---|---|---|---|---|
| `time_data_partition/` | 5,842 | 1,493 | 1,493 | `individual_time_samples` 的普通文件（不含 `_dwt`） |
| `time_data_partition/pretrain_index` | 38,116 | | | `extra_time_samples` 全部 32,274 + 上面的 train 5,842 |
| `time_data_partition_with_wind/` | 5,876 | 1,459 | 1,459 | `individual_time_samples_with_wind` |
| `data_partition/` | 5,863 | 1,472 | 1,472 | `individual_samples`（频域） |

- **val 和 test 是同一份**（`partition_data.py` 里直接 `val_samples = test_samples`），三组划分都是这样。
- 划分是**逐 clip 随机**的（约 80/20），test 里 12 段录音、两个节点全都有。同一段录音相邻的 clip 会分别落在 train 和 test 里，测出来的分数偏高。想做严格的泛化评估，可以按 recording 或 node 自己重新划分。
- `pretrain_index` 和 test 没有交集。
- `time_data_partition/` 里还有 6 个 noisy/1107 相关的文件（`noisy_test_index*.txt`、`test_index_1107_*.txt`），指向 `Parkland_Dirty`、`Parkland_1107` 两个数据集，这里没有那两份数据，所以 `sample_id` 全是 null。

## 注意事项

- 类别不平衡：主数据集里 polaris 2,782 vs warhog 1,606（按 14,670 计，含 `_dwt` 重复）。
- `individual_time_samples` 里每个 clip 出现两次（普通 + `_dwt`），统计和训练时别重复算。
- `individual_samples` 和其他 subset 有部分录音重叠（`Polaris0150pm`、`mustang0528` 等新录音在两边都有），混用时注意去重。
- 没有距离、速度信息；cs537aiot 配置里提到的 `Parkland_Distance_Speed` 不在本地。
