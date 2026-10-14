# Abdelzaher 组用这批数据的论文

这个文件夹收的是 Tarek Abdelzaher 组（UIUC）用 **声音 + 地面振动（acoustic + seismic）认车** 这条线的论文。本地 `datasets/` 里的 Parkland、GracesQuarters、ICT、M3N-VC 基本都是这个组自己采的，这些论文就是在这批数据上做出来的。

一共存了 17 篇 PDF。另外有 9 篇相关论文只在付费墙后面，没下载到，列在最后。

## 先说结论

### 论文里的名字 ↔ 本地文件夹

论文里很少直接说「GracesQuarters」「ICT」，一般用 MOD 之类的代号，或者干脆不写名字。下表是能对上的部分。**「确定」那几行是样本数和本地文件一模一样，我在 PDF 里亲自核对过。**

| 论文里叫 | 本地是 | 有多确定 | 证据 |
|---|---|---|---|
| **MOD**（Moving Object Detection） | `Parkland` | 确定 | FOCAL 写 39,609 个样本、7,335 个有标签 = `extra_time_samples` 32,274 + `individual_time_samples` 7,335 |
| VibroFM 的「MOD，38,116 个无标签样本」 | `Parkland/time_data_partition/pretrain_index.txt` | 确定 | 正好 38,116 行 |
| Dywave 的「MOD，8,828 个」 | `Parkland/time_data_partition` 的 train 5,842 + val 1,493 + test 1,493 | 确定 | 加起来正好 8,828（val 和 test 是同一份，被算了两次） |
| Dywave 的 **MOD-OOD**（4 类，35,162 个） | `GracesQuarters/day_1_sensor_all` + `day_2_sensor_all` | 确定 | 15,796 + 19,366 = 35,162 |
| VibroFM / FMSys 的 **Evaluation Set**（4 种车，两天，12,635 个） | `GracesQuarters/indices/GQ/train_index.txt` 里 day 1 的部分 | 确定 | 正好 12,635 个 |
| Micro-FM 的 Evaluation Set（polaris、warthog、皮卡，多车同时，有距离） | `GracesQuarters/2024-08-06-GQ`、`2024-08-07-GQ` | 很可能 | 车型、多车、GPS 距离都对得上 |
| **M3N-VC** | `M3N-VC` | 确定 | 名字一样。数据集是 RestoreML 这篇提出来的 |
| 四辆民用车（CX-30、GLE-350、MX-5/Miata、Mustang） | `ICT` / `M3N-VC` | 很可能 | Acies-OS、生成式增强（NeurIPS24）都是这四辆车 |
| West Point | `2026-04-14-West-Point` | — | **没有任何一篇论文用到**，是最新的数据 |

### 每个数据集被谁用过

- **Parkland（MOD）**：几乎所有自监督论文的标准测试集。FOCAL、FreqMAE、PhyMask、AdaTS、InfoMAE、Dywave、量化那篇、SudokuSens（实地数据）都用了。
- **GracesQuarters**：一般当「换个场地还行不行」的测试集。FMSys 案例研究、VibroFM、Micro-FM、Dywave（MOD-OOD）用了。IoBT-OS 用的是最早期的 GQ 数据。
- **ICT / M3N-VC**：比较新的工作在用。RestoreML（提出 M3N-VC）、SPAR、ZipFM、Acies-OS、生成式增强（NeurIPS24）。
- **West Point**：还没人用过。

### 这条线是怎么发展的（大白话版）

1. **2022 年前后**：先在 Graces Quarters 野外场地把系统搭起来（IoBT-OS），能在树莓派上实时认车。
2. **2023 年**：采了 Parkland（MOD）。核心问题是「标签太贵」，所以做**自监督预训练**：先用一大堆没标签的数据学会「听懂」振动信号，再用很少的标签微调。代表作是 FOCAL。
3. **2024 年**：把预训练模型叫成「**振动基础模型**」（VibroFM、Micro-FM），重点是换场地还好使、能跑在传感器节点上。同时改进预训练方法（FreqMAE、PhyMask），做数据增强（生成模型造数据），做部署平台（Acies-OS）。
4. **2025 年**：从「一个节点」走向「**一群节点**」。采了 M3N-VC（6 个场景、每个场景 6–8 个节点），做多节点联合建模（SPAR）、省带宽（ZipFM）、部署后自动纠错（RestoreML）。同时继续改进预训练（AdaTS、InfoMAE）。
5. **2026 年**：更聪明的「切 token」方式（Dywave），开放域数据增强（REMIX）。

## 看分数之前先知道这件事

**很多论文在 Parkland（MOD）上用的是逐样本随机划分（8:1:1）。** 同一段录音里前后挨着的 2 秒片段，会一个进训练集、一个进测试集，长得几乎一样，所以分数偏高：MOD 上动不动就 95% 以上。

真正能看出本事的是换场地的实验，比如在 Parkland 上训练、在 GracesQuarters 上测试。这时候分数会掉到 50–90%。读下面的数字时记住这一点。

---

## 逐篇解读

按时间排序。每篇都写了：一句话、干了啥、用了啥数据、效果怎么样。

### 1. SudokuSens（SenSys 2023）

- 文件：[2023-SenSys-SudokuSens.pdf](2023-SenSys-SudokuSens.pdf)
- 作者：Tianshi Wang, Jinyang Li, Ruijie Wang 等
- **一句话**：车的种类 × 环境的组合太多，采不过来，那就让模型自己「脑补」没采过的组合。
- **干了啥**：
  - 把「哪辆车」和「在什么环境」画成一张表格，就像数独：有的格子采到了，有的空着。
  - 用一个生成模型（条件 VAE）把「车的声音特征」和「环境的影响」拆开，再拼出空格子里的数据。这样采集量随车和环境的数量线性增长，而不是乘起来。
  - 另外加了一种对比学习，让模型不怕风之类的干扰。
- **用的数据**：
  - 量化实验主要用 ACIDS（陆军的公开车辆数据）和两个人体动作数据集。
  - 自己采的实地数据是 Parkland 的「大号版」：Polaris、Warthog、Silverado、摩托、Tesla、Mustang、MX-5 等 8 种车，屋顶、停车场、州立公园等 7 个场景。
- **效果**：比最好的基线高 10.7%–26.9% 准确率，表格越空提升越大。实地测试里，没采过的组合（Mustang 在碎石地、大风天）也能认。
- **缺点**：数据全齐的时候没优势，大部分结果只有图没有表。

### 2. FOCAL（NeurIPS 2023）⭐ 这条线的基石

- 文件：[2023-NeurIPS-FOCAL.pdf](2023-NeurIPS-FOCAL.pdf)，代码 github.com/tomoyoshki/focal
- 作者：Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu 等
- **一句话**：没标签的声音和振动数据，怎么让模型自己学到有用的特征。
- **干了啥**：
  - 以前的方法只学「声音和振动共同的信息」，比如车来了两边都变响。
  - FOCAL 把每个模态的特征拆成两半：**共享的一半**（两个模态都有的信息）和**私有的一半**（只有声音或只有振动才有的信息），两半互相垂直，不重复。
  - 再加一条规矩：时间上挨得近的片段，特征也该挨得近。
- **用的数据**：MOD（= 本地 Parkland，7 类，39,609 个样本里 7,335 个有标签）、ACIDS、RealWorld-HAR、PAMAP2。
- **效果**：在 MOD 上只训练一层线性分类器，准确率 **98.1%**。最好的基线（MoCo）是 93.9%，直接用全部标签做监督训练是 89.5%。总的来说比 11 个基线都强。
- **缺点**：MOD 是随机划分的，分数偏乐观；在 PAMAP2 上不如监督学习。

### 3. On the Efficiency and Robustness of Vibration-Based Foundation Models（FMSys 2024）

- 文件：[2024-FMSys-VibFMCaseStudy.pdf](2024-FMSys-VibFMCaseStudy.pdf)
- 作者：Tomoyoshi Kimura, Jinyang Li, Tianshi Wang 等
- **一句话**：把 FOCAL 拿到真实野外部署里检验：预训练模型换个地方还好不好使？
- **干了啥**：在 Parkland（MOD）上做 FOCAL 预训练，然后冻住模型，只在新场地数据上训练最后一层。和从零开始的监督训练比。
- **用的数据**：预训练用 MOD（Parkland）；测试用一个两天的新部署（= 本地 **GracesQuarters 老数据**），4 种目标（Polaris、Warthog、Husky、民用车）。第一天安静，第二天有人、发电机、风。
- **效果**（SW-T 模型，100% / 10% / 1% 标签）：
  - 第一天：FOCAL 95.3% / 94.3% / **83.1%**，监督 98.4% / 74.3% / **36.6%**。标签越少优势越大。
  - 第二天（用第一天训练的模型去测）：FOCAL 66.4%，监督 54.5%。
- **缺点**：标签全的时候监督学习还是赢；只有 3 页的 workshop 论文。

### 4. FreqMAE（WWW 2024）

- 文件：[2024-WWW-FreqMAE.pdf](2024-WWW-FreqMAE.pdf)
- 作者：Denizhan Kara, Tomoyoshi Kimura, Shengzhong Liu 等
- **一句话**：把「遮住一部分让模型猜」（MAE）这招改造得更适合振动信号。
- **干了啥**：
  - MAE 本来是图像上的方法：遮住一部分，让模型补回来，补着补着就学会了。
  - 改了三处：频率上的局部注意力；每个模态分私有/共享空间（像 FOCAL）；损失函数更看重低频和信号强的样本，因为车的特征主要在低频。
- **用的数据**：MOD（Parkland）、ACIDS、RealWorld-HAR、PAMAP2；还有 MOD-A/B/C 三个换位置、换车、换场地（屋顶）的测试集。
- **效果**：MOD 上 95.2%，比 CMC（90.5%）、原版 MAE（90.2%）都高；换场地测试也都比原版 MAE 好（例如 87.5% vs 85.8%）。树莓派 3 上处理一个 2 秒样本要 0.97 秒，原版 MAE 要 1.54 秒。
- **缺点**：在 PAMAP2 上不如监督学习。

### 5. Acies-OS（ICCCN 2024）

- 文件：[2024-ICCCN-AciesOS.pdf](2024-ICCCN-AciesOS.pdf)
- 作者：Jinyang Li, Yizhuo Chen, Tomoyoshi Kimura 等
- **一句话**：一个给野外传感器网络用的「操作系统」，能远程部署模型、换模型、自动处理节点挂掉。
- **干了啥**：每个节点的数据和模型都按名字组织，比如 `/节点3/seismic/分类结果`，用发布/订阅方式传。后台有个「数字孪生」：一份虚拟的系统副本，用来挑最合适的模型、测延迟、发现故障后自动切换。
- **用的数据**：实地部署，8 个树莓派节点（麦克风 + 地震计 + GPS），两个地点，4 辆车（2018 Mustang、2022 MX-5、2023 CX-30、2017 GLE-350）。这就是后来 **ICT / M3N-VC** 那批数据的来源。
- **效果**：
  - 有风的时候，系统自动选的「声音 + 振动」模型保持准确，只用声音的模型就崩了。
  - 1,800 秒里人为制造 4 次服务故障、2 次节点故障，全部自动恢复。
  - 结果都是图，没什么具体数字。

### 6. VibroFM（MASS 2024）

- 文件：[2024-MASS-VibroFM.pdf](2024-MASS-VibroFM.pdf)
- 作者：Tomoyoshi Kimura, Jinyang Li, Tianshi Wang 等
- **一句话**：提出「微型基础模型」（µFM）：不追求大，就专门做振动信号，小到能塞进传感器节点里。
- **干了啥**：
  - 第 3 篇的加长版：加了一个更轻的模型（TSMixer），加了用 3 倍数据预训练的 VibroFM-Large。
  - 测了没见过的车能不能认、距离能不能估，还在真实设备上测了速度和内存。
- **用的数据**：
  - 预训练：MOD（= Parkland `pretrain_index`，38,116 个，21 小时）。
  - VibroFM-Large 再加 4 辆民用车、3 个新场地，共 66 小时、119k 个样本（很可能是 ICT / M3N-VC 那批）。
  - 测试：GracesQuarters 老数据（12,635 个）。
- **效果**：
  - 1% 标签时，TSMixer 版 73.8% vs 监督 55.6%。
  - 预训练里没有 Husky，照样能认出来。
  - 在 RaspberryShake 4D（树莓派 4）上，每 2 秒的数据推理不到 0.2 秒；模型 45 MB，轻量版 7.5 MB。
  - 微调时内存 35 MB，监督训练要 858 MB。
  - VibroFM-Large 在所有标签比例下都比普通版好，说明数据越多越好。
- **缺点**：作者自己说证据是「anecdotal」（个例性的）。

### 7. The Case for Micro Foundation Models（IEEE CogMI/CIC 2024）

- 文件：[2024-CIC-MicroFM.pdf](2024-CIC-MicroFM.pdf)
- 作者：Tomoyoshi Kimura 等（含 Shenoy、Srivastava、Wigness、Abdelzaher）
- **一句话**：一个预训练好的振动模型当「底座」，上面挂一堆小插件，分别做认车、估距离、认多辆车。
- **干了啥**：
  - VibroFM 冻住不动，每个任务只训练一个很小的头（一层线性层或者 4 层 MLP）。
  - 加任务几乎不增加成本；一个节点坏了，别的节点换个小头就能顶上它的活。
- **用的数据**：训练用之前的单车数据；测试用新采的碎石路场地，Polaris、Warthog、皮卡，有单车也有多车，带 GPS 距离（= 本地 **GracesQuarters 2024-08-06/07**）。
- **效果**：
  - 新场地认车约 90%，监督模型明显掉分（具体数只在图里）。
  - 距离分三档（≤15 / 15–30 / ≥40 m），MLP 头最好；距离回归误差比监督模型小，但 40 m 以外变差。
  - 底座推理 188 ms，每个小头不到 1.5 ms、不到 0.52 MB。

### 8. PhyMask（SenSys 2024）

- 文件：[2024-SenSys-PhyMask.pdf](2024-SenSys-PhyMask.pdf)
- 作者：Denizhan Kara, Tomoyoshi Kimura 等
- **一句话**：MAE 遮哪里不该随便遮，应该按物理规律挑着遮。
- **干了啥**：
  - 原版 MAE 随机遮住一块块信号。PhyMask 先算每块的能量高不高、和前后连不连贯，重要的块更可能被留下来，不重要的大胆遮掉。
  - 能遮到 96%，省算力也省数据。
- **用的数据**：MOD（Parkland）、ACIDS、PAMAP2、RealWorld-HAR；还有一个实地案例：200×50 m 的碎石/沙地，8 个树莓派节点，4 辆车。论文没写名字，可能是 ICT 或 GQ。
- **效果**：
  - 平均准确率 87.9%，原版 MAE 是 80.9%。
  - 预训练数据少用 75%，内存少用 14%。
  - 实地测试：全部标签时 >90%，10% 标签时 85%。
  - 不过在 MOD 上 89.3%，反而不如 CMC 的 90.5%。
- **缺点**：自动版（B-PhyMask）在 MOD 上遮得太狠（99%），掉了约 10%。

### 9. Fine-Grained Control of Generative Data Augmentation（NeurIPS 2024）

- 文件：[2024-NeurIPS-FineGrainedGenAug.pdf](2024-NeurIPS-FineGrainedGenAug.pdf)
- 作者：Tianshi Wang, Qikai Yang, Ruijie Wang 等
- **一句话**：用生成模型造训练数据，而且能精确控制造出来的数据「长什么样」。
- **干了啥**：
  - 先算出每段信号的统计指标，比如有多「乱」、频谱多「平」。
  - 训练一个扩散模型，输入「标签 + 这些指标」就生成对应的信号。
  - 想要新数据就改指标：比如把「乱度」调高一点，模拟换到碎石地上。
- **用的数据**：自采的车辆数据，城市停车场（水泥）、州立公园（碎石）、乡村土路，5 个节点，4 种车：紧凑 SUV、中型 SUV、跑车、敞篷跑车。这很可能就是 **ICT / M3N-VC** 那 4 辆车。另外还有 RealWorld-HAR 和脑电数据。
- **效果**：认车 82.0%，不增强是 78.8%，传统增强 80.2%。只用水泥地数据训练，调指标模拟碎石地后，迁移效果变好。
- **缺点**：论文说「5 种车」但只列了 4 种（第 5 类可能是背景）；那些指标是人手挑的。

### 10. RestoreML（DCOSS-IoT 2025，最佳论文）⭐ M3N-VC 出处

- 文件：[2025-DCOSS-IoT-RestoreML.pdf](2025-DCOSS-IoT-RestoreML.pdf)，数据 Zenodo 15215210 / github.com/restoreml/m3n-vc
- 作者：Jinyang Li, Yizhuo Chen, Ruijie Wang 等（Hongjue Zhao 也在作者里）
- **一句话**：模型部署到野外以后，环境变了、节点换了，准确率就掉。能不能不要人标数据，让它自己修好？
- **干了啥**：每个节点各自预测，汇总到中间，在一小段时间和空间里取平均。平均出来的结果通常比单个节点准，就把它当「老师」，反过来微调每个节点（集体蒸馏）。不靠谱的节点（总和大家唱反调的）不参与平均。
- **用的数据**：**提出了 M3N-VC**：18.26 小时、31 GB、6 个场景（H08 晴天、H24 雨天、S31 土路、A06 柏油、I29 大风、I22 两车同时），每个场景 6–8 个节点，4 辆车（CX-30、GLE-350、Mustang、MX-5）。另外有一个室内声音事件数据集。
- **效果**（Macro-F1）：
  - H24 训练、S31 测试，振动模型：平均 10.2 → 45.4，比「拿真标签微调」的 44.8 还高。
  - 声音模型：84.8 → 91.0；多节点合起来：89.7 → 94.4。
  - I22 上用 VibroFM：72.6 → 79.3，常用方法 TENT 只有 55.4。
- **缺点**：只用振动的模型本身就弱（上限 45–61 F1）；有的场景提升很小。

### 11. Input-Aware Quantization（INFOCOM 2025）

- 文件：[2025-INFOCOM-InputAwareQuant.pdf](2025-INFOCOM-InputAwareQuant.pdf)
- 作者：Ashitabh Misra, Nurani Saoda, Tarek Abdelzaher
- **一句话**：在比树莓派还小的单片机（Raspberry Pi Pico）上认车，而且保证不超时。
- **干了啥**：
  - 模型压缩（量化）的位数一般是固定的。这篇按输入难易来调：简单的输入用低精度跑得快，难的用高精度。
  - 有严格的延迟和内存上限，保证一次都不超。
- **用的数据**：MOD（Parkland），**只用声音**，0.5 秒片段，类别合并成 4 类（摩托、Silverado、Polaris、其他）。
- **效果**：准确率比最好的同类方法高 4.6%，比自适应量化基线最多高 19%；**0 次超时**，对手最多 27.6% 超时。ResNet18 约 85–89%。

### 12. ZipFM（MASS 2025）

- 文件：[2025-MASS-ZipFM.pdf](2025-MASS-ZipFM.pdf)
- 作者：Tianchen Wang, Yizhuo Chen, Hongjue Zhao 等
- **一句话**：一堆节点要把数据传给中心的大模型，网络扛不住，那就压缩。但压多狠、怎么压，要聪明地选。
- **干了啥**：每个节点、每个模态、每个时刻自动选压缩方法和压缩程度。判断压坏了没有不用标签，看大模型的内部特征被压缩改变了多少。
- **用的数据**：**M3N-VC**（H24 做单车定位，I22 做多车检测）、一个地震数据集（Ridgecrest）、RealWorld-HAR。
- **效果**：只传原始数据量的约 **4%**，定位误差 4.19 m，几乎没变差；多车检测精度也基本不变。
- **缺点**：和基线的对比基本都是曲线图。

### 13. SPAR（arXiv 2025，预印本）

- 文件：[2025-arXiv-SPAR.pdf](2025-arXiv-SPAR.pdf)
- 作者：Yizhuo Chen, Tianchen Wang, You Lyu 等
- **一句话**：多个节点一起看一辆车时，让模型知道「每个节点在哪」。
- **干了啥**：
  - 在 MAE 预训练里加入每个节点的位置坐标，并做随机旋转、平移，让模型学的是相对位置。
  - 遮住一部分信号让它猜的同时，也遮住某个节点的位置让它猜。
  - 这样模型能利用「同一辆车在不同位置的节点上听起来不一样」这个信息。
- **用的数据**：**M3N-VC**（H24 分类和定位，I22 多车检测，留出 H08/H24 测没见过的节点摆放），还有 Ridgecrest 地震数据、RealWorld-HAR。
- **效果**：
  - 单车定位误差 4.20 m，最好的基线 5.06 m。
  - **多车检测 mAP@6m 71.8，基线 52.1**，提升很大。
  - 认车 99.3% vs 99.1%，单场景认车已经到天花板了。
- **缺点**：还是预印本；检测标签来自手机 GPS（1 Hz，有噪声）。

### 14. AdaTS（NeurIPS 2025）

- 文件：[2025-NeurIPS-AdaTS.pdf](2025-NeurIPS-AdaTS.pdf)
- 作者：Denizhan Kara, Tomoyoshi Kimura, Jinyang Li 等（Hongjue Zhao 也在作者里）
- **一句话**：对比学习里「两个样本像不像」不该是非黑即白，应该有程度之分。
- **干了啥**：
  - 一个插件，能加到 FOCAL、TS2Vec、TS-TCC 这些方法上。
  - 用时频相关性衡量两个片段多像，然后要求模型保持「谁比谁更像」的排序。
  - 对变化快的信号，时间上的相似度衰减也更快。
- **用的数据**：MOD（Parkland，认车 7 类，另有距离 4 类、速度 4 类）、ACIDS、PAMAP2、RealWorld-HAR，还有 UCR/UEA 两个公开时间序列库。
- **效果**：
  - FOCAL + AdaTS 在 MOD 上 97.1%，原版 FOCAL 是 95.5%。
  - 加到 TS2Vec 上提升最大：平均 68.2% → 81.9%。
- **缺点**：MOD 还是随机划分；在车辆数据上提升比人体动作数据小。

### 15. InfoMAE（WWW 2025）

- 文件：[2025-WWW-InfoMAE.pdf](2025-WWW-InfoMAE.pdf)
- 作者：Tomoyoshi Kimura, Xinlin Li, Osama Hanna 等
- **一句话**：同步采集的「声音 + 振动」成对数据很少，但单独的声音或单独的振动数据很多。能不能分开预训练，再用很少的配对数据「粘」起来？
- **干了啥**：声音模型、振动模型各自用 MAE 预训练，然后用信息论的损失把两个特征空间对齐，只需要少量同步配对样本。
- **用的数据**：MOD（Parkland）为主，另有两个没写名字的自采场地：一个在「户外研究设施附近的州立公园」（可能是 GQ），一个是 4 辆车（两台 SUV、跑车、肌肉车，可能是 ICT 那批）。另外还有 PAMAP2、RealWorld-HAR。
- **效果**：只用 5% 的配对数据就能到 88.3%；直接联合训练只有 33.3%，FOCAL 86.9%。跨场地对齐在 5 种组合里全部胜出。

### 16. Foundation Models for CPS-IoT（arXiv 2025，综述）

- 文件：[2025-arXiv-FMforCPSIoT.pdf](2025-arXiv-FMforCPSIoT.pdf)
- 作者：Ozan Baris Mulayim, Yizhuo Chen 等（多个学校合写）
- **一句话**：综述：基础模型用到物联网上有哪些机会和难点。
- **相关部分**：第 3.2 节有一个小实验，6 个树莓派节点、民用车、GPS，像 M3N-VC。用多节点 MAE（SPAR 的前身）做认车和跟踪，比 FOCAL 等方法好，但只有柱状图没有数字。
- **价值**：了解大方向可以看，具体技术看 SPAR。

### 17. Dywave（ICML 2026）

- 文件：[2026-ICML-Dywave.pdf](2026-ICML-Dywave.pdf)
- 作者：Tomoyoshi Kimura, Denizhan Kara, Jinyang Li 等（Hongjue Zhao 也在作者里）
- **一句话**：Transformer 处理信号要先切成一小块一小块（token），固定长度地切太笨了，应该按「事件」来切。
- **干了啥**：
  - 用小波变换找到信号里的「关键时刻」（比如车刚开过来），在那里多切几刀、切细一点；安静的地方合并成一大块。
  - 这样 token 数量是动态的，又少又不会把一个事件切碎。
- **用的数据**：MOD（= Parkland，8,828 个）、**MOD-OOD（= GracesQuarters 老数据，35,162 个）**、PAMAP2、RealWorld-HAR、Ego4D IMU、WESAD。
- **效果**：
  - MOD 振动 80.8% vs PatchTST 77.0%。
  - **换场地（MOD-OOD）只训练最后一层：振动 75.7% vs PatchTST 34.1%**，差距巨大。
  - token 最多少 75%；信号越长越省（30 秒的片段 35 个 token vs 372 个）。
- **缺点**：2 秒的短片段没什么可省的，在树莓派上预处理的开销还抵不回来。

---

## 没下载到的（付费墙）

这些都只有 IEEE/ACM 的付费版，或者网站拦了自动下载。能用学校账号的话，可以手动下。

| 论文 | 会议 | 跟这批数据的关系 | 备注 |
|---|---|---|---|
| IoBT-OS | ICCCN 2022 | 早期 GQ 数据（Polaris、Silverado、Warthog，115 分钟） | Dongxin Liu 的博士论文第 6 章有相关内容：DeepSense + 自监督 91.2%，20 秒投票到 99% |
| The Methodological Pitfall of Dataset-Driven Research | MILCOM 2022 | 很可能是 Parkland/GQ | 讲「测试集上好、部署就崩」，和上面说的随机划分问题直接相关，**值得一读** |
| DiffPhys | ICCCN 2025 | 认车、测速、测距 | 用可微分物理模型（衰减、多普勒、散射）做数据增强，号称最多 +9% |
| DynaGen | INFOCOM 2025 | 声音 + 振动认车 | 条件扩散模型造数据 |
| OpenMAE | IMWUT 2025 | 很可能是 MOD | 用 500 万+ 开放世界 RaspberryShake 振动数据预训练。ACM 上免费，但网站拦了脚本：[链接](https://dl.acm.org/doi/pdf/10.1145/3729485) |
| SemiCMT | IMWUT 2025 | 不确定 | 从声音向振动迁移知识。ACM 上免费：[链接](https://dl.acm.org/doi/pdf/10.1145/3699779) |
| Towards Acies-OS 2.0 | IEEE AIoT 2025 | 不确定 | Acies-OS 的升级版 |
| Agentic LLM-Assisted Edge AI | IEEE CogMI 2025 | 只拿车辆数据举例 | 愿景论文：让 LLM 智能体调度边缘 AI |
| REMIX | ICCCN 2026 | 不确定 | 开放域信号混合做数据增强，可能和 OpenMAE 一脉相承 |

## 查过但没收的

MMBind（SenSys 2025）、FusionBridge（SenSys 2026）、Tracking without Seeing 等：用的是人体动作、相机、激光雷达或模拟数据，不是这批振动数据。
