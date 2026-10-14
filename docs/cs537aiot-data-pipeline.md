# cs537aiot 是怎么得到训练数据的

这份文档讲 `cs537aiot/`（FoundationSense 课程项目，FOCAL 那一系列代码）从原始录音一路走到「喂进模型的 tensor」的全过程，以及里面的坑。

**范围**：仓库里只有 **Parkland** 的预处理脚本（`src/data_preprocess/Parkland/`）。GQ、ICT 只是在配置里引用了别处做好的索引文件，怎么生成的在这个仓库里看不到。

下面的路径都相对于 `cs537aiot/src/`。

## 一张图看懂

```
原始录音 CSV（16 kHz 麦克风 + 100 Hz 地震计，每段录音 × 每个节点）
   │  ① 掐头去尾（手工表 data_trunk.py）
   │  ② 可选：麦克风叠加风噪
   │  ③ 麦克风 16 kHz → 8 kHz
   │  ④ 切成 2 秒一段，不重叠；每段再分 10 个 0.2 秒小格
   ▼
每个样本一个 .pt 文件
   ├─ individual_time_samples/  （rs3 / rs1，带标签的主数据集）
   ├─ extra_time_samples/       （其他节点 + 新车，用来预训练）
   └─ *_freq_samples/           （同样的样本先做了 FFT，训练时根本不用）
   │  ⑤ 写 txt 名单：逐个文件随机 80/20 分 train/test，val = test
   ▼
train_index.txt / val_index.txt / test_index.txt / pretrain_index.txt
   │  ⑥ 训练时：按名单 torch.load → 可选只取一部分标签
   │  ⑦ 数据增强（看训练阶段）
   │  ⑧ 每个 0.2 秒小格现场做 FFT，实部/虚部当 2 个通道
   ▼
模型输入：audio [B, 2, 10, 1600]，seismic [B, 2, 10, 20]
```

## 第一步：离线切样本

脚本：`data_preprocess/Parkland/extract_samples.py`（主数据集）、`extract_pretrain_samples.py`（预训练数据）。

### ① 原始录音

每段录音一个文件夹，名字就是录音名，下面每个节点一个子文件夹：

```
raw_data/<录音名>/<节点>/
    aud16000.csv 或 aud.csv   麦克风，16000 Hz（多列时取第 0 列）
    ehz.csv                   地震计，100 Hz
```

录音名开头就是类别，比如 `Polaris0150pm`、`mustang0528`、`walk2`。`folder_to_label()` 用「录音名里包含哪个类名」来定标签。

标签编号（`LABELS`）：

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Polaris | Warhog | Silverado | motor | tesla | mustang | walk | bicycle | forester | pickup | scooter |

### ② 掐头去尾（`data_trunk.py`）

录音开头和结尾一般是人在摆设备、车还没来。`start_time_shift` / `end_time_shift` 两张表**手工写**了每段录音、每个节点开头和结尾各砍多少秒，比如 `mustang` 的 rs1 开头砍 380 秒、结尾砍 30 秒。Polaris、Warhog、Silverado 这几段是 0，不砍。

### ③ 可选：叠加风噪

`add_wind=True` 时，用 `wind_noise_generator.py`（Nelke & Vary 2014 的风噪模型，参数在 `wind_noise_model_v1.1.mat`）生成一段「阵风」，**乘以 80000** 后加到 16 kHz 的麦克风数据上，地震计数据不动。`individual_time_samples_with_wind/` 就是这样生成的。

### ④ 降采样 + 切片

1. 麦克风用 torchaudio 从 16000 Hz 重采样到 8000 Hz（`AUD_DOWNSAMPLE_RATE = 2`），地震计保持 100 Hz。
2. 每 2 秒切一段（`SEGMENT_SPAN = 2`），不重叠（`SEGMENT_OVERLAP_RATIO = 0`），最后不满 2 秒的扔掉。
3. 每段再分成 10 个 0.2 秒的小格（`INTERVAL_SPAN = 0.2`）。

所以一个样本的形状是：

| 模态 | 形状 | 含义 |
|---|---|---|
| audio | `[1, 10, 1600]` | 1 个通道 × 10 个小格 × 每格 1600 个点（0.2 s × 8000 Hz） |
| seismic | `[1, 10, 20]` | 1 个通道 × 10 个小格 × 每格 20 个点（0.2 s × 100 Hz） |

代码里有一个按地震计标准差过滤安静片段的逻辑，但阈值 `STD_THRESHOLD = 0`，等于没开。

### ⑤ 存两份

`process_one_sample()` 给每个 2 秒样本存两个文件，文件名都是 `<录音名>_<节点>_<序号>.pt`：

```python
{
    "label": tensor(id),  # int64
    "flag": {"shake": {"audio": True, "seismic": True}},
    "data": {"shake": {"audio": ..., "seismic": ...}},
}
```

- **时域版** → `individual_time_samples/`，`data` 里是上面的波形。
- **频域版** → `individual_freq_samples/`，对每个 0.2 秒小格做 FFT，实部、虚部分开存，形状 `[2, 10, 1600]` / `[2, 10, 20]`。

**训练时根本不读频域版**，FFT 是在训练时现算的（见下面第 ⑧ 步）。这也是 unified 版本没转换 `*_freq_samples` 的原因。

### ⑥ 关键：只挑一个节点进主数据集

**带标签的主数据集**（`extract_samples.py`）只收 12 段「干净」录音（`PRESERVED_CLEAN_FOLDERS`），每次车经过**只用一个节点**：

| 录音 | 用哪个节点 |
|---|---|
| `motor`、`mustang0528`、`walk2`、`tesla` | 只用 `rs3`（`SUBJECTS = {"rs3"}`） |
| `Polaris*`、`Warhog*`、`Silverado*`（代码里叫 `GC_FOLDERS`，是另一个场地录的） | 只用 `rs1` |

**预训练数据**（`extract_pretrain_samples.py`，`PRESERVED_EXTRA_FOLDERS`）收的是：

- 上面四段录音**其他节点**（rs1、rs2、rs7）录到的数据；
- 新录音的全部节点（rs1、rs2、rs3、rs7）：`bicycle`、`forester`、`pickup`、`scooter`、`motor2`、`walk` 等。

这些都放进 `extra_time_samples/`。

## 第二步：写 train/test 名单

脚本：`partition_data.py`、`partition_data_pretrain.py`。数据文件本身不动，只生成 txt，每行一个 `.pt` 的完整路径。

### 监督名单（`partition_data.py`）

1. 只要 `PRESERVED_FOLDERS` 里的录音，以及指定节点（`shakes = ["rs1", "rs2", "rs3", "rs7"]`）的样本。
2. **逐个文件扔骰子**：`random.random() < 0.8` 进 train，否则进 test。
3. 只保留两个模态 `flag` 都为 True 的样本。
4. **`val_samples = test_samples`**：验证集和测试集是同一份。
5. 写出 `train_index.txt`、`val_index.txt`、`test_index.txt`，再 rsync 到另一台机器（`eugene`）。

### 预训练名单（`partition_data_pretrain.py`）

`pretrain_index.txt` = `extra_time_samples/` 里所有两个模态都齐全的样本 + 监督的 `train_index.txt`（逐行复制）。

本地 `datasets/Parkland/time_data_partition/` 里的就是这些文件：train 5,842、val = test 1,493、pretrain 38,116（= 32,274 + 5,842）。

## 第三步：训练时怎么读

### ⑦ 选名单、读样本

`input_utils/multi_modal_dataloader.py` 的 `create_dataloader()`：

| 阶段 | 读哪个名单（来自 `data/Parkland.yaml`） |
|---|---|
| 自监督预训练（`stage=pretrain`） | `pretrain_index_file` |
| 监督训练 / 微调的训练集 | `<任务>.<前缀>train_index_file` |
| 验证 / 测试 | `<任务>.<前缀>val_index_file` / `test_index_file` |

- 「前缀」来自命令行的 tag。比如 finetune tag 以 `gq1` 开头，就去读 `gq1_train_index_file`。前缀里有 `gq` 时，类别数会被强行改成 4。
- 配置里还有 `gq2_*`、`gqday1_*`、`gqday2_*`（指向 2024-08-06/07 GQ）、`ict_*`（ICTMixed）这些名单，路径都在别的机器上。

`input_utils/multi_modal_dataset.py` 的 `MultiModalDataset`：

1. 用 `np.loadtxt` 读名单。
2. `label_ratio < 1` 时（少标签实验，比如 0.01），把名单打乱，只留前 1%。
3. 每次 `torch.load` 一个 `.pt`，取出 `data` 和 `label`：
   - `label` 是 dict（ACIDS）时，按任务取 `vehicle_type` / `distance` / `speed` / `terrain`。
   - `label` 是字符串（ICT 那批）时，查一张硬编码的表：`gle350 → 7`、`miata → 8`、`cx30 → 9`、`mustang → 5`。
   - 二维的 numpy 数组（比如 ICT 的 `[10, 1600]`）前面补一个维度变成 `[1, 10, 1600]`。
4. 可选类别平衡采样（`balanced_sample`），只对 ACIDS 和 `Parkland_Miata` 开。

还有一个 `MultiModalSequenceDataset`：把同一段录音里连续的样本按序号排好，每 `seq_len = 4` 个打成一组，给 TS2Vec、TNC、TSTCC、InfoMAE 这类需要「时间上相邻」信息的预训练方法用。

### ⑧ 数据增强 + 现场 FFT

`data_augmenter/Augmenter.py` 的 `forward()`：

1. **统一形状**：不到 4 维的输入 reshape 成 `[B, 1, num_segments=10, -1]`。
2. **数据增强**，看训练阶段：

   | 阶段 | 调用 | 做什么（以 DeepSense 的配置为例） |
   |---|---|---|
   | 监督训练 | `forward("fixed")` | 按配置固定做，比如时域 mixup：两个样本按比例混合，标签也混 |
   | 自监督预训练 | `forward("random")` | 每个 batch 从增强池里随机挑一种，比如相位偏移（phase shift） |
   | 微调 / 测试 / KNN 评估 | `forward("no")` | 不增强 |

3. **现场 FFT**（`fft_preprocess`）：沿最后一维（每个 0.2 秒小格）做 `torch.fft.fft`，实部、虚部拆成 2 个通道：
   - audio `[B, 1, 10, 1600]` → `[B, 2, 10, 1600]`
   - seismic `[B, 1, 10, 20]` → `[B, 2, 10, 20]`
4. 喂给模型（`models/TransformerV4.py`、DeepSense 等）。

**没有做任何数值归一化。** `input_utils/normalize.py` 里有 `normalize_input()`（除以每个模态的最大绝对值），但整个代码库没有一处调用它。模型输入就是原始幅度的 FFT，靠模型里的 LayerNorm（Swin 的 `patch_norm`）兜底。`train.py` 里的 `count_range()` 只是统计一下数值范围，不改数据。

## 坑

1. **随机划分，分数偏高。** 名单是逐个 2 秒片段扔骰子分的，同一段录音里相邻的片段，一个在 train、一个在 test，长得几乎一样。论文里 MOD 上 95% 以上的分数要打折看。
2. **val 和 test 是同一份。** 拿 val 挑模型、再在 test 上报分数，等于在测试集上挑模型。
3. **预训练和测试间接重叠。** 测试集里是 rs3 看到的某次经过，同一次经过被 rs1、rs2、rs7 录到的数据在 `extra_time_samples`（预训练集）里。预训练不用标签，但模型等于提前见过这些事件。
4. **标签编号撞车。** 字符串标签的映射 `gle350 → 7`、`miata → 8`、`cx30 → 9`，和 Parkland 的 bicycle 7、forester 8、pickup 9 撞了。GQ 老数据的 husky 存的是 3，和 motor 的 3 撞了。所以不同数据集只能分开训练、各自定类别数，不能直接混在一起。
5. **2024 年 GQ 的多车标签读不了。** 那批数据的标签是数组（比如 `["warthog", "polaris"]`），`MultiModalDataset` 没处理这种情况。
6. **掐头去尾是手工的。** 换一批新录音就得重新人工定每个节点砍多少秒，没有自动检测「车来了没有」。
7. **风噪强度写死。** 风噪统一乘 80000，没有按信号能量调 SNR，不同录音加完风之后信噪比差别很大。

## 跟我们 `datasets_unified/` 的关系

模型吃的是「时域波形 → 每 0.2 秒一格 → FFT」。unified 里存的是压平后的波形，比如 `[16000]`、`[200]`：

```python
import h5py
import torch

with h5py.File("datasets_unified/Parkland/individual_time_samples/signals.h5") as h5:
    # 和 cs537aiot 读到的一模一样
    audio = torch.from_numpy(h5["signals/audio"][0]).reshape(1, 10, 1600)
    seismic = torch.from_numpy(h5["signals/seismic"][0]).reshape(1, 10, 20)
```

所以要复现 FOCAL 这一系列，可以直接用 `datasets_unified`：

- 名单用 `datasets_unified/splits/Parkland/time_data_partition/*.parquet`，里面的 `sample_id` 已经解析到本地样本。
- Parkland 的 `individual_time_samples` 里还有 `_dwt` 文件（波形和普通文件一模一样），名单里没有它们，按名单取就不会混进来。
- 其他数据集的采样率和 dtype 不一样（例如 M3N-VC 是 1600 / 200 Hz、int16 / int32），reshape 之前要先确认每格的点数。
