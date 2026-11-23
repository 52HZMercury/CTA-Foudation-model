# Merlin × Valve 实验日志 (v0.11 · Merlin 栅格)

> **v0.11**（2026-11-21）：**E2 三折跑完**（并集 Dice + AdaBN + 新解码器）：val MRE
> **2.89 ± 0.38 mm**、SDR@5 **0.924 ± 0.012**，总耗时 4.87 h（§12.3）。比 E0 旧掩蔽的
> 3.89 ± 0.24 mm 好约 1.0 mm；但 E0 必须用并集 Dice 重跑后才算同协议对照。
>
> **v0.10**（2026-11-20）：**E2 的 20 mm 平台定位到损失缺陷并修复**（§12.2）。`SoftDiceLoss`
> 的 Dice 前景掩蔽由逐通道改为**所有通道 GT 前景的并集**；fold 0 的 40 epoch 探针从 19.96 → **3.22 mm**。
> 旧掩蔽下 E0 的 3.89 mm 需要用新损失重跑才可比（约 10 h）。
>
> **v0.9**（2026-11-20）：**E0 三折跑完**，结果记入 §12；文件从"实验计划"改名为"实验日志"，
> 此后每次实验的结果都追加到 §12。E0 基线：val MRE **3.89 ± 0.24 mm**、SDR@5 **0.894 ± 0.025**。
>
> **v0.8**（2026-11-19）：**暂停预训练数据的栅格重建**。
> 1. 按当前方案先不建 1864（E10）与 2790（E3 自监督）的裁剪数据：管线已停，已生成的 1444 例转换标注、
>    预训练中心/分割已删除，`cases.json` 回到 420 例下游。
> 2. 下游 420 例的 Merlin 栅格数据（datalist、norm、QA）保留可用，E0/E2 可直接开跑。
> 3. 需要 E10 或 E3 时，先按 §8 重跑 convert-pretrain（+ `--include-unlabeled`）/ localize / crop，再继续。
>
> **v0.7**（2026-11-19）：必做改成「预训练池自监督 → 下游 GT 微调」两段式。
> 1. 新增必做 **E3**：Merlin 编码器先在 **2790 例预训练图像**上做掩码重建自监督（零标签），再用同一套编码器做 3 折下游微调。
> 2. 三条必做臂（E0/E2/E3）的下游微调集统一为 **420 例下游 GT**（同一折、同一 recipe、同一验证集）；
>    1444 例转换标注退出必做，降级为条件实验 E10（420 vs 1864）。
> 3. E9（剩余 1346 例继续自监督）被 E3 吸收，删除；数据管线需再补 1346 例无标注病例的 localize/crop。
> 4. 微调集回到 420 后按完整 epoch 训练；`--max-iters 30` 只留给 E10 的 1864 臂。
>
> **v0.6**（2026-11-19）：
> 1. 预训练池实测覆盖：2790 例里 **1444 例可监督**（803 例 5 点齐全 + 641 例只有 LCA/RCA 开口），
>    已转成下游 gt 格式并入训练集；训练池 420 → **1864**，每折 train 1567/1571/1574。
>    剩余 1346 例无标注，只留给 E9。Marker=窦底已用下游 GT 验证。
> 2. 训练集变大后 epoch 不能按全量过一遍算：统一 `--max-iters 30`（每 epoch 120 样本）。
> 3. 新增条件实验 **E10 数据消融**（420 vs 1864），回答"转换标注值多少"。
>
> **v0.5**：只有一种栅格，就叫 **Merlin 栅格**（1.5×1.5×3.0 mm）。A/Merlin 栅格的说法全部废弃。
> **v0.4 的改动**：
> 1. 既然是在 Merlin 上微调，栅格直接用 Merlin 原生的 1.5×1.5×3.0 mm。
> 2. 实验数从 11 个压到 **必做 2 个 + 条件触发 3 个**。
> 数据仍然从 release 原始 `image.nii.gz` 开始，不走 `/data/disk_c/valve_pipeline`。

---

## 0. 一页结论

1. 数据分工：**420 例下游 GT 做微调**；**2790 例预训练图像（2219 fct + 571 dicom）做自监督**，两者患者 ID 零重叠，全部从 release 原始图像开始处理。
2. 栅格只有一种：**Merlin 栅格**，1.5×1.5×3.0 mm，224×224×160（336×336×480 mm FOV）。
3. 必做回答两个问题，三条臂同一套 420 微调数据、同一套折、同一套 recipe：
   - 主问题：**Merlin 预训练权重 + 微调，比同栅格从零训练的 SegResNetDS 好吗？** 分母 E0（scratch），分子 E2（Merlin）；
   - 第二问：**在 2790 例上再做一段域内自监督，值不值？** E3（Merlin + 自监督）vs E2。
4. 其余实验按结果触发，不预先烧卡。
5. 裁剪中心已定（TotalSegmentator 心脏质心 + 群体偏移）；1444 例转换标注不进必做，只作为 E10 数据消融的监督数据。
   1864 / 2790 的裁剪数据**暂不生成**（2026-11-19 已停止管线并删除半成品），E10 / E3 开跑前再重建。carry 外部验证按原计划。
6. **E0 已完成**（2026-11-20，§12.1）：3 折 val MRE **3.89 ± 0.24 mm**、SDR@5 **0.894 ± 0.025**，
   合计 9.75 h（三折都在 300 epoch 前早停）。E2 的对照分母已就位。
7. **E2 已完成**（2026-11-21，§12.3）：并集 Dice 下 3 折 val MRE **2.89 ± 0.38 mm**、SDR@5 **0.924 ± 0.012**，
   合计 4.87 h。与 E0 旧掩蔽数字比低约 1.0 mm，但 E0 用并集 Dice 重跑后才是同协议对照。

---

## 1. 数据实测（已核对，不是转述文档）

### 1.1 图像

| 项 | huaxi (190) | carry (230) |
|---|---|---|
| 方向 | LPS × 420（全部） | 同 |
| 平面 spacing | 0.309 – 0.885 mm（中位 0.509） | 0.289 – 0.977（中位 0.428） |
| 层厚 | 0.30 – 0.70 mm（中位 0.625） | **0.166 – 1.50**（中位 0.5） |
| 层数 | 214 – 1112（中位 256） | 128 – 1568（中位 292） |

**关键**：release 里是**原始 CT 栅格**，spacing 各不相同，必须自己重采样到 B 的 1.5×1.5×3.0 mm。

### 1.2 标注覆盖（`gt/` 下实际文件数）

| 文件 | huaxi /190 | carry /230 |
|---|---|---|
| `L.mrk.json` | 188 | 224 |
| `R.mrk.json` | 174 | 216 |
| `N.mrk.json` | 166 | 188 |
| `LCA.mrk.json` | 190 | 226 |
| `RCA.mrk.json` | 188 | 225 |
| `annulus.mrk.json` | 188 | 217 |
| `root.nrrd` | 177 | **160** |
| `calcification.nrrd` | 177 | 226 |

标注是 3D Slicer Fiducial，LPS，世界坐标 mm。缺标注很普遍（N 只有 166/190），所以**每个任务都必须带 per-case existence mask**，损失里按 mask 屏蔽，指标也按 mask 统计。

### 1.3 预训练数据

- `pretrain/huaxi/fct_pid_*` 2219 例：`image.nii.gz` + `raw_fct_marker/`
- `pretrain/huaxi/dicom_pid_*` 571 例：只有 `image.nii.gz`
- 与下游 420 例 **患者 ID 零重叠**（`convert-pretrain` 每次校验）。

标注覆盖（`raw_fct_marker/` 实际文件，逐例核对）：

| 可用信息 | 例数 | 处理 |
|---|---|---|
| 3 个窦底 Marker + LCA/RCA 开口 + 瓣环 | **803** | 转成 5 通道 GT，仅 E10 用 |
| 只有 LCA/RCA 开口（含 428 例无 Marker） | **641** | 只登记 2 个通道，existence 其余为 0，仅 E10 用 |
| 只有瓣环 / 不完整 Marker | 107 | 不转标注，只进 E3 自监督池 |
| 无任何标注 | 668（fct）+ 571（dicom） | 只进 E3 自监督池 |

**E3 自监督池 = 全部 2790 例**（不看标注）；1444 例转换标注只在条件实验 E10（420 vs 1864 微调）里当监督信号。

**Marker = 窦底**已验证（2026-11-19）：
1. Marker_1..3 到瓣环平面距离 p90 0.2 mm，与下游 GT 的 L/R/N 同构（下游三窦点到平面 p50 0.1 mm）；
2. 窦周径平面在 Marker 上方 7.5–11.1 mm（窦底而非交界的高度）；
3. 分配规则「L = 角度上离 LCA 最近，R = 离 RCA 最近」在 314 例全标注下游 GT 上命中 99.4% / 98.7%，取 6 种排列全局最优。

只有 2 个 Marker 的 210 例**不做推导**：分配角 p90 达 56°/128°，缺的那颗常是 N，强行分配会灌错误 L/R 标签。

转换产物：`work/prep/pretrain_gt/<dirname>/gt/*.mrk.json`（下游格式），登记为 `cases.json` 里 `split="pretrain"` 的病例，走同一条 localize/crop/targets 管线；split 时只进每一折的 train，不进任何验证集，且只有 E10 的 1864 臂会用到。无标注的 1346 例登记为 `split="pretrain_ssl"`，只跑 localize/crop（不跑 targets/qa），也不进任何 datalist——E3 的数据加载器直接读这些裁剪图。

---

## 2. 数据处理流程 `valve_prep/`

产物写 `~/program/CTA-Foudation-model/work/prep/merlin/`，不改动 release。

```
release_v1 ──index──> cases.json
           ──convert-pretrain──> pretrain_gt/<dirname>/gt/*.mrk.json（1444 例，E10 用）
           ──convert-pretrain --include-unlabeled──> 1346 例无标注登记（E3 用，只裁剪）
           ──crop──>   merlin/<id>/image_crop.nii.gz + root/calc mask + crop_meta.json
           ──targets──> merlin/<id>/targets/*_heatmap.nii.gz, annulus_ring.nii.gz, plane.json
           ──qa──>     qa_report.csv + overlays/
           ──split──>  merlin/datalist.json
```

### 2.1 栅格参数

| 参数 | 值 |
|---|---|
| 目标 spacing | 1.5 × 1.5 × 3.0 mm |
| 输出尺寸 | 224 × 224 × 160 |
| 物理 FOV | 336 × 336 × 480 mm |
| 裁剪中心 | TotalSegmentator 心脏质心 + 群体偏移，**再 +16 mm 右 +16 mm 上** |
| 插值 | 图像 linear，mask nearest |
| 越界 | 常数填充 −1000 HU |
| HU 归一化 | **固定 [−1000, 1000]**，即 Merlin 原生，不用统计值 |

### 2.2 现状

| 项 | 状态 |
|---|---|
| 管线代码 `valve_prep/` | ✅ 完成（index / convert-pretrain / localize / crop / targets / qa / split） |
| 0.5 mm 试跑数据 | ✅ 曾跑通（417 例、2085 张热图、零越界）——只验证过管线，不是实验数据 |
| 预训练标注转换 | ⏸ 2026-11-19 已删除（脚本仍在）；需要 E10 时重跑 `convert-pretrain`：1444 例登记（803 例 5 通道 + 641 例只有 LCA/RCA），与下游 PID 零重叠 |
| **Merlin 栅格数据（420 下游）** | ✅ TotalSegmentator 中心重建 `work/prep/merlin/`：420 例子集、3 折 123/67、127/63、130/60、外部 230、QA 零越界 |
| **Merlin 栅格数据（1864，E10 用）** | ⏸ 已取消（2026-11-19）：localize 到 264/1864 时停止，预训练中心/转换标注已删除；需要时按 §8.4 重建 |
| **Merlin 栅格数据（2790，E3 用）** | ⏸ 暂不生成：E3 开跑前登记 1346 例（`pretrain_ssl`）并补 localize/crop（连同 1444 例一起重跑） |
| 裁剪中心 | ✅ TotalSegmentator `total` 模型 whole-heart 质心 + 群体偏移（−5.1, −1.1, +5.6 mm） |
| 旧 GT 中心数据 | 保留在 `work/prep/merlin_gtcenter/`（仅作对照，训练不用） |
| **E0 训练（3 折）** | ✅ 2026-11-20 完成：早停于 144/169/134 epoch，合计 9.75 h，结果见 §12.1 |

---

## 3. 网络

```
CT patch (B,1,224,224,160)
  → Merlin I3ResNet(return_skips=True, BN frozen)
       skip1..4 →  UNet 解码器（InstanceNorm）
                →  任务头 5ch / 6ch / 1ch
```

- 编码器 122 M（含两个头，不用），解码器 ~12–15 M
- skip 尺寸 224×224×160 → 112×112×80、56×56×40…，解码器参数与尺寸无关
- 三任务分开训练

---

## 4. 实验矩阵（3 + 4）

### 必做

| ID | 编码器 | 初始化 | 预训练阶段 | 下游微调集 | 折数 | 回答什么 |
|---|---|---|---|---|---|---|
| **E0** | SegResNetDS | scratch | — | 420 下游 GT | 3 | 同栅格从零训练的基线（分母） |
| **E2** | Merlin | Merlin 权重 | — | 420 下游 GT | 3 | 预训练权重值不值（分子） |
| **E3** | Merlin | Merlin 权重 | **2790 例掩码重建自监督**（一次，与折无关） | 420 下游 GT | 3 | 域内自监督再适配值不值 |

三条臂必须跑满 3 折、同一套 recipe、同一套 fold 划分、**同一套 420 例微调数据**。
自监督阶段只看预训练池图像（2790 例、零标签），不碰任何下游图像；E3 的 3 折共用同一份自监督编码器权重。
任何一折的 val 与外部集都只有下游人工标注（无标签噪声、无患者重叠）。

### 条件触发（满足条件才跑，每个先只跑 fold 0）

| ID | 触发条件 | 内容 |
|---|---|---|
| E1 | E2 没赢过 E0 | Merlin 冻结编码器——区分"权重没用"和"微调没调好" |
| E5 | 获胜的 Merlin 臂（E2/E3）赢过 E0 且差距 > 1 mm | 钙化分割，从获胜臂初始化，看增益能否迁移 |
| E4 | 同上 | 瓣环平面回归，从获胜臂初始化 |
| **E10** | 出主结论后（用户关心数据来源的贡献） | **数据消融**：在获胜臂上把下游微调集从 420 换成 **1864**（420 下游 + 1444 转换标注），先 fold 0。开跑前先重建 1864 数据（§8.4）。回答"1444 例转换标注到底值多少"；子问题可把 1444 换成 803（去掉只有开口的 641 例），看 2 通道样本有没有拖后腿 |

**砍掉的**：E1b（部分解冻）、E2b（lr 5e-5）、E6（大 FOV 解耦）、E7（归一化消融）、spacing 对比（另一种栅格已取消）。这些都没有改变主结论的能力，只是把 3 天的事拖成 3 周。
E3 这个编号原先预留给 spacing 对比，v0.7 起改用于域内自监督；E9 已删除。

---

## 5. 训练配方

```yaml
optimizer: AdamW
lr: head+decoder 2e-4 | encoder 1e-5
scheduler: cosine, warmup 10 epoch
weight_decay: 1e-5
amp: bf16
batch: 4（224×224×160）
epochs: 300，early stop patience 50（按 val MRE）
max_iters: 必做臂不加（420 例就是完整 epoch）；E10 的 1864 臂加 `--max-iters 30`，见 §7
loss: MSE + Dice + Wing，逐点乘 existence mask
aug: rotate ±0.15 rad, translate ±5 voxel, scale ±10%, gauss σ=0.05, p=0.3
     ⚠ 禁止 flip
bn: encoder 全程 eval()
norm: 图像固定 [−1000, 1000] → [0, 1]，同 Merlin
```

缺失标注：热图通道和损失**都乘 existence mask**；某例缺 N 时该通道 GT 全 0、损失权重 0。
641 例预训练只有 LCA/RCA，三窦通道的 existence=0（损失跳过），不是 -1（压假峰）——
它们是"未标注"，不是"解剖上不存在"。（这条只影响 E10 的 1864 臂。）

### 5.1 E3 自监督配方（一次性，与折无关）

```yaml
objective: 3D 掩码重建（SimMIM 式，只在被遮体素上算损失）
mask: 16×16×16 block，每例随机遮 60%
input: 同 Merlin 栅格、同 [−1000,1000] → [0,1] 归一化
encoder: Merlin I3ResNet（release 权重）初始化，return_skips=True
decoder: 同下游 U-Net 解码器（InstanceNorm），输出 1 通道；自监督结束后丢弃
loss: L1 on masked voxels
optimizer: AdamW, enc lr 1e-5 / dec 2e-4, bf16, batch 4
budget: 100 iter/epoch × 30 epoch ≈ 3000 iter；按 E2 实测 ~1.1 s/iter ≈ **1 h**
        （原 11 h 按旧解码器速度估的，作废；首次正式跑后回填实测）
output: 只存编码器权重 → E3 的 3 折微调初始化
```

**实现**：`valve_train/ssl.py`（2026-11-23 写完并冒烟：掩码自检 + 64³ 合成张量的前向/损失/
`encoder.pth` → `build_encoder` 严格加载闭环）。重建头重新初始化（零权重 + bias −4 是热图专用的）；
编码器 BN 开 AdaBN；`encoder.pth` 的键带 `encode_image.i3_resnet.` 前缀，直接给 `--merlin-ckpt`。

```bash
bash run_ssl.sh --out work/ssl/e3
# 3 折微调：
bash run_train.sh --grid merlin --fold N --arch merlin --adabn --merlin-ckpt work/ssl/e3/encoder.pth
```

选掩码重建而不是对比学习：不用负样本、不受 batch 大小限制（单卡 batch 4），
解码器本来就要为 E2 写、直接复用。若 30 epoch 后下游没提升，再试 SimSiam（同样无负样本）。

### 5.2 E2 实现（`valve_train/merlin_net.py`）

```yaml
encoder: Merlin I3ResNet (ResNet-152 膨胀), return_skips=True
         权重 = release checkpoint 的 encode_image.i3_resnet.*，strict 加载
         BN affine 冻结；默认 running stats 也用 Merlin 的（eval），E2 实测必须开 --adabn
decoder: 照 MONAI SegResNetDS（E0 用的就是它）：反卷积上采样、skip 1×1 投影后相加、残差块
         宽度 256/128/64/32/16；Z 只降 4 次（160→10），最后一级只在 X/Y 上 2×
head:    1×1×1 conv, weight 零初始化 + bias −4（热图极稀疏，sigmoid(0)=0.5 的背景压不下去；
         随机初始化还会让通道 sigmoid 饱和死掉）
lr:      encoder 1e-5, decoder + head 2e-4（其余同 §5）
params:  编码器 121.7 M + 解码器 3.1 M = 124.9 M
speed:   ~35 s/epoch（fold 0 实测，bf16 + encoder checkpointing），GPU 峰值 ~32 GB
```

损失：`SoftDiceLoss` 的前景掩蔽改为**本样本所有通道 GT 前景的并集**。按通道各自掩蔽时，
通道 j 在别的标志点上起的假峰完全不进 Dice（MSE/AWL 全图平均也压不住），实测 E2 的假峰
77.8% 落在同病例另一个标志点上，val 全局 MRE 卡在 20 mm；改并集后同一折 ep34 就降到 3.22 mm。
E0 的历史结果用的是旧掩蔽，**必须用新损失重跑才可比**（§12.2）。

方向对齐：下游数据是 **LPS**，Merlin 预处理是 **RAS**；forward 里翻转 X/Y 两轴送编码器，输出翻回。
不对齐时 fold 0 第 5 个 epoch val MRE 57.5 mm，对齐后 41.6 mm（同期 E0 47.2 mm）。

---

## 6. 评估

- 3 折 mean ± std；carry 227 例作外部集，报 bootstrap 95% CI
- **val 折未变**：split 的随机数只作用于 huaxi 190 例，加预训练例不改变折划分，
  所以"420 例训练 vs 1864 例训练"（E10）是同一验证集上的干净消融
- 与对照做**逐例配对 Wilcoxon**，p<0.05 才算改进
- Landmark：MRE、SDR@5/10 mm，按点、按 mask 统计
- **分层厚报告**：Merlin 栅格 z 方向 3 mm，carry 里原始层厚 0.17–1.5 mm 的病例要分开看
- **分轴报告**：平面内误差 (xy) 与穿层误差 (z) 分开。z 方向体素 3 mm，穿层误差的理论下限比平面内高，混在一起看不出问题在哪
- Plane：法向夹角、中心距离；Calcification：Dice、HD95、连通域检出率
- 推理：3 折 ensemble，峰值用**亚体素质心**

> ⚠ **Merlin 栅格的精度下限**：3 mm 层厚下，SDR@2 mm 会明显低于 0.5 mm 栅格。这是栅格决定的，不是模型差。
> 主指标用 **SDR@5 mm**，@2 mm 仅作参考。E0 与 E2 在同栅格上比较，这个限制不影响两者谁赢。

---

## 7. 算力

机器：单卡 RTX PRO 6000 Blackwell 96 GB。
Merlin 栅格（224×224×160，batch 4）实测 **76–83 s/epoch**（E0 三折，每折 123–130 例，一个完整 pass ≈ 31 iter），跑满 300 epoch = 6.3 h/折；E0 实际早停只用了一半。

必做臂的微调集就是 420 下游，按完整 epoch 训练即可，不需要截断。
只有 E10 的 1864 臂（每折 1567–1574 例，完整过一遍约 16 min，300 epoch = 80 h/折）必须加 `--max-iters 30`
（每 epoch 120 样本），否则预算翻 13 倍。

| 阶段 | 估计 |
|---|---|
| Merlin 栅格数据生成（420 已建；1444 + 1346 需要时再建） | 420 见 §2.2；预训练部分约 3–4 h（localize 为主） |
| E0 3 折 | ✅ 实测 **9.75 h**（3.04/3.69/3.02 h；早停在 144/169/134 epoch，patience 50 生效；跑满 300 epoch 才是 19 h） |
| E2 3 折 | ✅ 实测 **4.87 h**（2.32/0.94/1.61 h；35–37 s/epoch，早停在 219/84/139 epoch；跑满 300 epoch 才是 8.7 h） |
| E0 3 折重跑（并集 Dice，§12.2） | **待跑**，同旧值 ~10 h（76–83 s/epoch） |
| E3 自监督（一次） | ~1 h（3000 iter × E2 实测 ~1.1 s/iter；首次跑后回填） |
| E3 3 折 | ~5–10 h（同 E2 实测 4.87 h） |
| 条件触发的单折实验 | 每个 5–10 h |

必做部分约 **34–40 h ≈ 1.5–2 天**（含 E0 重跑；E3 自监督 ~1 h 为估计值）。条件实验按需再排。

---

## 8. 实施清单

```
valve_prep/
├── common.py        # 路径、mrk.json 解析、世界↔体素转换、GRIDS 参数
└── prepare.py       # 子命令 index / convert-pretrain / localize / crop / targets / qa / split

valve_train/
├── data.py          # 数据集、GPU 合成热图、几何增强、亚体素峰值（demo() 自检）
├── losses.py        # MSE + SoftDice + AdaptiveWing（三态 existence mask）
├── ssl.py           # E3 掩码重建自监督（mask + 训练循环），输出编码器权重
└── train.py         # SegResNetDS / --arch merlin 训练、验证、早停、TensorBoard；--adabn
```

已跑通：旧 0.5 mm 栅格上 E0 fold 0 冒烟 3 epoch，loss 2.19→0.37，val MRE 61→14.7 mm，50 s/epoch。
Merlin 栅格数据（420 例下游）已跑通训练栈自检；**E10 的 1864 例和 E3 的 2790 例自监督都还没训练过**。

**下一步（按顺序）**：
1. ~~目录改名 + 刷新 datalist~~ ✅ 已完成：`work/prep/merlin/`，datalist 路径已刷新
2. ~~抽查 6 例~~ ✅ 已抽查（新中心下 max_err 0.44–0.72 voxel，5 点全部在窦部）
3. ~~预训练标注转换 + 验证~~ ✅ 803+641 例，规则在下游 GT 上 99.4%/98.7%
4. ~~重建 1864 例数据~~ ⏸ 已停止（E10 暂缓）；需要时 localize → crop → targets → qa → split 重跑
5. ~~登记剩余 1346 例~~ ⏸ 随 E3 一起做（`convert-pretrain --include-unlabeled` + localize/crop）
6. ~~冒烟 1 epoch + 实测 epoch 时间，更新 §7~~ ✅ 76–83 s/epoch，§7 已更新
7. ~~接 Merlin 编码器（`--arch merlin`）~~ ✅ `valve_train/merlin_net.py`；~~`ssl.py`~~ ✅ 已写并冒烟（§5.1），等 2790 数据
8. ~~E0 3 折~~ ✅（§12.1）；~~E2 3 折~~ ✅ 2.89 ± 0.38 mm（§12.3）→ **E0 用并集 Dice 重跑 3 折**（§12.2）；E3 开跑前先重建 2790 数据，再自监督 → 3 折

---

## 9. 排期

| 周 | 交付 | 卡点 |
|---|---|---|
| W1 | 420 例数据就绪；预训练重建暂停（保留代码，需要时再跑） | — |
| W2 | ~~E0 3 折~~ ✅（3.89 mm）+ 接 Merlin / 写 ssl.py | — |
| W3–4 | ~~E2 各 3 折~~ ✅ 4.9 h（并集 Dice）+ E0 重跑 3 折 | E0 重跑 ~10 h，同卡串行 |
| W4–5 | E3：先重建 2790 数据（约半天）→ 自监督（10–15 h）→ 3 折 | 自监督 loss 要降 |
| W5 末 | 出结论：Merlin vs scratch vs 域内自监督 | 条件实验按结论触发（E10 数据消融排在最前） |

---

## 10. 风险

| 风险 | 对策 |
|---|---|
| 3 mm 层厚限制绝对精度 | 主指标改 SDR@5；分轴报告；E0/E2 同栅格比较不受影响 |
| TotalSegmentator 失败或中心偏移大 | 实测 420/420 成功、零越界；心脏质心到瓣环中位 11 mm、p95 22 mm |
| `heartchambers_highres` / `aortic_sinuses` 需 license | 改用免费 `total` 的 whole-heart 类；群体偏移一次校准，逐例零 GT |
| 1444 例预训练标签是转换来的，与下游标注协议可能有细微差异 | 已在 314 例下游 GT 上验证分配规则（99.4%/98.7%）；预训练例只进 train，验证/外部集全是人工标注 |
| 641 例只有 LCA/RCA 通道 | existence mask 置 0 的通道在损失里跳过（0=跳过，不是 -1 压峰），不会教模型"此处无窦点" |
| carry 层厚差异大（0.166–1.5 mm） | 外部验证按层厚分层报告 |
| root/calc 标注 carry 只有 160/230 | 钙化任务样本量比想象小，先查清缺的是哪批 |
| 掩码重建 30 epoch 不足以适配 Merlin 权重 | 自监督是一次性的 10–15 h，不涨就换 SimSiam 或加 epoch，不牵动 3 折预算 |
| 新管线与旧结果不可比 | 明确写进结论，不与历史数字比，只与 E0 比 |

---

## 11. 待确认

1. ~~P1 裁剪中心用 TotalSegmentator 还是 GT 质心？~~ ✅ 已解决：TotalSegmentator `total`（whole-heart）+ 群体偏移，全部数据已重建
2. huaxi 是否留出独立 test 集，还是 190 例全部进 3 折？
3. ~~2790 例预训练是否纳入~~ ✅ 已定（v0.7）：2790 例全部用于 E3 自监督；1444 例转换标注只在 E10 数据消融里当监督信号；必做微调集是 420 下游 GT。

---

## 12. 实验记录

### 12.1 E0 — SegResNetDS 从零训练（基线，2026-11-20 完成）

**配置**：SegResNetDS **87.2 M**（init_filters 32、blocks_down 1,2,2,4,4、dsdepth 1、InstanceNorm），
AdamW lr 2e-4 / wd 1e-5、warmup 10 + cosine、batch 4、bf16；
loss = MSE + SoftDice + AdaptiveWing（逐点乘 existence mask）；增强 rotate ±0.15 rad / translate ±5 vox / scale ±10% / gauss σ0.05，p=0.3，禁 flip。
300 epoch 上限、每 5 epoch 验一次、patience 50，按 val MRE 存 `best.pth`。数据 = 420 下游 GT 的 Merlin 栅格，3 折。

启动方式：`bash run_train.sh --grid merlin --fold N`（每次新建 `logs/<时间戳>/`，stdout 覆盖到项目根 `out.log`）。

| 折 | run 目录 | train/val | 早停 epoch | 最佳 epoch | MRE (mm) | SDR@2 | SDR@5 | SDR@10 | 时长 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `logs/20261119_203539` | 123/67 | 144 | 94 | 4.166 | 0.568 | 0.867 | 0.921 | 3.04 h |
| 1 | `logs/20261119_234540` | 127/63 | 169 | 119 | 3.765 | 0.609 | 0.897 | 0.914 | 3.69 h |
| 2 | `logs/20261120_033802` | 130/60 | 134 | 84 | 3.745 | 0.644 | 0.917 | 0.934 | 3.02 h |
| **均值 ± std** | | | | | **3.89 ± 0.24** | **0.607 ± 0.038** | **0.894 ± 0.025** | **0.923 ± 0.010** | **9.75 h** |

逐点 MRE（mm，只在有标注的点上统计）：

| 折 | L | R | N | LCA | RCA |
|---|---|---|---|---|---|
| 0 | 5.222 | 3.973 | 2.935 | 4.352 | 4.195 |
| 1 | 3.889 | 6.029 | 3.637 | 1.503 | 3.932 |
| 2 | 2.684 | 4.740 | 2.770 | 5.237 | 3.246 |
| **均值 ± std** | **3.93 ± 1.27** | **4.91 ± 1.04** | **3.11 ± 0.46** | **3.70 ± 1.95** | **3.79 ± 0.49** |

**观察**
- 三折都在 300 epoch 前早停（最佳 epoch 84–119，patience 50 生效），最后一次 lr 停在 cosine 中段的 0.85–1.23e-4；满 300 epoch 的预算减半，§7 已按实测更新。
- epoch 76–83 s，与 §7 的 76 s 估计一致。
- 折间波动主要来自 L / R / LCA 的个别病例（fold 1 的 R 6.03 mm、fold 2 的 LCA 5.24 mm）；N 最稳（3.11 ± 0.46）。
- 冒烟跑（`logs/20261119_203454`，1 epoch × 3 iter）MRE 119.6 mm，只验证管线，不算结果。
- 尚未做：3 折 ensemble 推理、carry 外部集、分轴（xy/z）与分层厚报告。这些等 E2 出结果后两臂同协议一起算；目前只有 3 折 val 单模型数字。
- **E0 是后续所有比较的分母**：E2 / E3 用同一 420 例、同一折、同一 recipe，只在编码器初始化上不同。

---

### 12.2 E2 — 排查记录：val MRE 卡在 ~20 mm（2026-11-20）

**症状**：E2 的所有变体（冻结 BN / AdaBN / encoder lr 1e-5 vs 2e-4 / 随机编码器对照 /
两版解码器）在 40 epoch 内 val MRE 都停在 **17.6–21 mm**，同期 E0 已到 6.4 mm。
训练 loss 却和 E0 相近（ep19 0.17 vs 0.14），说明不是"没学到东西"。

**定位过程**

| 检查 | 结果 | 结论 |
|---|---|---|
| ckpt 编码器权重 vs 原始 Merlin | conv 权重变化 1–3%，BN 统计 10–80% | 梯度通，编码器在学 |
| 2 例过拟合（Merlin） | loss 0.012 但 MRE 20 mm | loss 与定位脱钩 |
| 2 例过拟合（E0 架构） | loss 0.034 但 MRE 22 mm | 不是 Merlin 特有 |
| GT 热图自己过 `subvoxel_peak` | 0.04–0.10 mm | 目标/峰值提取没问题 |
| 2 例过拟合逐通道 argmax | **5 个通道都落在同一个点**（同病例另一个标志点），prob@gt 0.67–0.89 | 通道纠缠 + 假峰 |
| val 315 点：预测点离谁最近 | **77.8% 更靠近另一个标志点**，90.8% 在某个标志点 5 mm 内 | 跨通道泄漏 |
| val 全局 vs GT 附近局部 argmax | 全局 19.96 mm，局部(R≤12) **3.80 mm** | **热图基本正确，假峰抢了全局最大值** |
| 同指标 E0 | 全局 4.17 / 局部 2.55 | E0 没有这个问题 |

**根因**：`SoftDiceLoss` 把预测按**每个通道自己的 GT 前景**掩蔽后再算 Dice，通道 j 在别的
标志点上起的假峰完全不进损失；MSE 和 AWL 都是全图平均（800 万体素里 ~800 个），惩罚只有 ~1e-4。
于是"每个通道都放一个略高于真峰的假峰"几乎免费。

**修复**：Dice 前景掩蔽改成**本样本所有通道 GT 前景的并集**（`losses.py` 一处）。同一 fold 0：

| | ep4 | ep9 | ep14 | ep19 | ep34 | 最好 |
|---|---|---|---|---|---|---|
| 旧掩蔽（40 epoch） | 21.65 | 23.93 | 22.26 | 20.96 | 20.34 | 19.96 @29 |
| 并集掩蔽（40 epoch） | 25.35 | 12.24 | 3.68 | 3.88 | **3.22** | **3.22 @34**（仍在降） |
| E0（300 epoch 调度，旧掩蔽） | 47.2 | 16.6 | 14.3 | 7.9 | 6.5 | 4.17 @94 |

并集掩蔽下 2 例过拟合从 20 mm → **0.08 mm**；val 全局/局部 3.22/3.11，落在别的标志点的比例
77.8% → 1.3%；逐点 L 2.69 / R 3.18 / N 2.69 / LCA 3.13 / RCA 4.37。

**其它修复（保留）**
- head weight 零初始化 + bias −4：随机初始化时 5 个通道里有通道 sigmoid 饱和死掉（max prob 0.04）。
- `--adabn`：编码器 BN 保持 train 模式用 CTA 数据重估 running stats；冻结 BN 明显更差。
- 解码器照 SegResNetDS 重写（反卷积 + skip 相加 + 残差块），3.1 M，比旧 concat 解码器快一倍、省 2.6× 显存。

**协议后果**：E0 的 3.89 mm 是旧掩蔽下得到的，**必须用并集 Dice 重跑 3 折**才能和 E2 比较。
E2 正式跑：`--arch merlin --adabn`，300 epoch 调度、patience 50，其余同 E0；三折已完成，结果见 §12.3。

---

### 12.3 E2 — Merlin 编码器 + 解码器 3 折（2026-11-21 完成）

**配置**：MerlinUNet = I3ResNet（ResNet-152 膨胀，Merlin release 权重，121.7 M，AdaBN）
+ SegResNetDS 式解码器（3.1 M，反卷积 + skip 相加 + 残差块），合计 124.9 M；head 零初始化 + bias −4。
AdamW 解码器 lr 2e-4 / 编码器 lr 1e-5，wd 1e-5、warmup 10 + cosine。增强、loss 权重、batch 4、bf16、
每 5 epoch 验一次、patience 50 全部同 E0；loss 用并集前景 Dice（§12.2）。

启动方式：`bash run_train.sh --grid merlin --fold N --arch merlin --adabn`。

| 折 | run 目录 | train/val | 早停 epoch | 最佳 epoch | MRE (mm) | SDR@2 | SDR@5 | SDR@10 | 时长 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `logs/20261120_235633` | 123/67 | 219 | 169 | 2.862 | 0.448 | 0.914 | 0.975 | 2.32 h |
| 1 | `logs/20261121_021601` | 127/63 | 84 | 34 | 3.274 | 0.497 | 0.921 | 0.970 | 0.94 h |
| 2 | `logs/20261121_031215` | 130/60 | 139 | 89 | 2.521 | 0.502 | 0.938 | 0.976 | 1.61 h |
| **均值 ± std** | | | | | **2.89 ± 0.38** | **0.482 ± 0.030** | **0.924 ± 0.012** | **0.974 ± 0.003** | **4.87 h** |

逐点 MRE（mm）：

| 折 | L | R | N | LCA | RCA |
|---|---|---|---|---|---|
| 0 | 2.19 | 3.05 | 2.62 | 2.49 | 3.95 |
| 1 | 2.62 | 3.02 | 2.33 | 2.31 | 5.95 |
| 2 | 2.96 | 2.39 | 1.92 | 2.84 | 2.42 |
| **均值 ± std** | **2.59 ± 0.39** | **2.82 ± 0.37** | **2.29 ± 0.35** | **2.55 ± 0.27** | **4.11 ± 1.77** |

**与 E0 的对照（E0 数字是旧掩蔽，非同协议，仅作参考）**

| 臂 | MRE (mm) | SDR@5 | 总时长 |
|---|---|---|---|
| E0（旧掩蔽，§12.1） | 3.89 ± 0.24 | 0.894 ± 0.025 | 9.75 h |
| E2（并集 Dice，本节） | 2.89 ± 0.38 | 0.924 ± 0.012 | 4.87 h |
| E0（并集 Dice） | 待跑，§7 | | ~10 h |

**观察**
- 三折的最佳 epoch 全部优于 E0 同折（2.862 vs 4.166、3.274 vs 3.765、2.521 vs 3.745），RCA 之外的点集中 2.2–3.0 mm。
- fold 1 在 ep34 触底后 50 epoch 内不再改善（早停 84），RCA 5.95 mm 是全场最差的单点；RCA 折间方差也最大（4.11 ± 1.77）。
- fold 0 与 §12.2 的 40 epoch 探针一致（探针 3.22 @34），但完整调度又降到了 2.86 @169——探针低估了长调度。
- 35–37 s/epoch，逐 epoch 比 E0 快一倍多；三折 4.87 h，E0 同量级要 9.75 h。
- 结论以 E0 并集 Dice 重跑后的数字为准；目前只能说 E2 在旧 E0 数字上领先约 1.0 mm。

---

## 附：环境

```
~/anaconda3                      # Anaconda3-2025.12-1
conda activate ctafdm            # python 3.12, torch 2.9.1+cu128, monai 1.6.2, SimpleITK, nibabel,
                                 # TotalSegmentator 2.18.0
```

- TotalSegmentator 依赖的 nnunetv2 新版声明 `torch!=2.9.*`，会顺带把 torch 换成 CUDA 13 版。装法是
  `nnunetv2==2.6.4 --no-deps` + `torchvision==0.24.1 timm --no-deps`，保住 torch 2.9.1。
- `heartchambers_highres`、`aortic_sinuses` 在 2.18 是 license 模型；免费只有 `total`。

- Blackwell（sm_120）必须 cu128 及以上
- `pypi.org` / `download.pytorch.org` 直连不通，走 tuna（已写进 `~/.config/pip/pip.conf`）
- conda 通道已指向 tuna 的 `pkgs/main` + `pkgs/r`；但 tuna 的 `/anaconda/archive/` 有反爬，装 Anaconda 本体要用 `repo.anaconda.com`
- release 数据 420 例下游 + 2790 例预训练全部在 `/media/dell/.../valve_release_v1_20261113`
