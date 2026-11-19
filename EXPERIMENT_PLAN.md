# Merlin × Valve 微调实验计划 (v0.1)

> 目标：用 Merlin (CT 视觉-语言基础模型) 的图像编码器替换/增强现有 `valve_models` 里从零训练的
> SegResNetDS，验证其在主动脉瓣标志点 / 瓣环平面 / 钙化分割三个下游任务上的增益，并给出可复现的对照实验。

---

## 0. 一句话路线

**A 主线**：Merlin I3D-ResNet152 编码器 + U-Net 解码器（多任务头），**0.5 mm / 192×192×144** 输入，
与现有基线**只差 backbone**，直接可比。
**B 对照**：Merlin 原生 **1.5×1.5×3 mm / 224×224×160** 大 FOV 输入（匹配预训练分布）。
**C 可选**：用 2790 例 pretrain 做领域继续预训练（SSL / FCT marker 监督），再微调。

---

## 1. 已核实的现状

### 1.1 数据（`valve_release_v1_20261113`）

| 集合 | 数量 | 用途 |
|---|---|---|
| downstream/huaxi | 190 | 内部 3-fold CV + 测试 |
| downstream/carry | 230 | 外部验证 |
| pretrain/huaxi (fct_) | 2219 | 预训练（有好坏点/曲线/测量，**无报告文本**） |
| pretrain/huaxi (dicom_) | 571 | 预训练（只有图像） |

旧管线产物 `/data/disk_c/valve_pipeline/`（由 `hx-dell-1011-ssh` 持有，对 `hx-dell-dev` **只读**）：

- `splits.json`：`huaxi_dev 151 / huaxi_test 38 / carry_external 223 / excluded 8`
- `datalist_landmark.json`、`datalist_calcification.json`、`fingerprint.json`
- 每例 `crop/`：`image_crop.nii.gz`、`root_crop`、`calcification_crop`、`aorta_crop`、
  `landmarks_crop.json`、`plane_gt.json`、`heatmaps/*_heatmap_s3.0.nii.gz`

### 1.2 现有基线（要超越的对象）

| 任务 | 模型 | 输入 | 输出 | 关键超参 |
|---|---|---|---|---|
| Landmark | SegResNetDS(32, [1,2,2,4,4]) | 0.5 mm, 192×192×144 | 5ch 高斯热图 (σ=3 mm) | lr 2e-4, 300 ep, 3-fold, MSE+Dice+Wing |
| Plane | 同上 + 第 6 通道 annulus ring | 同上 | ring → SVD 拟合平面 | 从 landmark ckpt 初始化, encoder_lr 1e-5 |
| Calcification | SegResNetDS, 1ch | 同上, clip [-100, 2000] | 二值 mask | global+local Tversky, 300 ep |

crop 定义：主动脉根部质心 **右移 16 mm、上移 16 mm** 为裁剪中心，物理尺寸 **96×96×72 mm**。
统一按 **RAS** 方向，热图标签用 bilinear 插值（禁止 flip 增强）。

### 1.3 Merlin 的输入约束（与上面**不一致**，是本次最大工程点）

`Merlin(ImageEmbedding=True)` 的官方预处理 (`merlin/data/monai_transforms.py`)：

```text
Orientationd RAS → Spacingd (1.5, 1.5, 3.0) → [−1000, 1000] → [0,1]
→ SpatialPadd / CenterCrop 224×224×160
```

- 编码器：I3D-ResNet152（~60 M 参数），`I3ResNet.return_skips=True` **已支持返回 5 个尺度的特征**，
  可直接当 U-Net encoder 用（无需改 Merlin 源码）。
- 输入张量布局是 `(B, 1, H, W, D)`，forward 内部复制成 3 通道再 permute 到 `(B,3,D,H,W)` —— 接自己的
  dataloader 时容易踩。
- 全部是 **BatchNorm**，预训练 batch=18；我们微调 batch 只有 1–4，**必须冻结 BN 统计量**（`.eval()` +
  `requires_grad=False`），否则训练不稳定。
- 文本分支（Clinical-Longformer）本数据集**用不上**（无报告），只取图像编码器。

---

## 2. 数据适配方案

| 方案 | 输入 | 来源 | 优点 | 缺点 |
|---|---|---|---|---|
| **A（主线）** | 0.5 mm, 192×192×144 | 复用现有 crop + 热图 | 与基线完全同分布同 GT，唯一变量是 backbone；GT 现成 | 与 Merlin 预训练 spacing 差 3× |
| **B（对照）** | 1.5×1.5×3 mm, 224×224×160 | 从原始 `image.nii.gz` 重裁大 FOV（约 336×336×480 mm，实际≈全胸腹） | 完全匹配预训练分布 | 需重建 GT 热图（σ 用 mm，重采样到 1.5 mm 后 σ≈2 体素）；3 mm 层厚限制定位上限 |
| C（消融） | 0.5 mm, 96×96×72 mm 物理 FOV | 同 A | 排除 FOV 影响 | — |

实现要点：

1. A 方案只用 image + 现成 5 通道热图，`StackHeatmapsd` 逻辑可原样复用（见 `valve_models/landmark/dataset.py`），
   新增一个 `MerlinDataset` 适配 `(B,1,H,W,D)` 布局即可。
2. HU 归一化：沿用 `fingerprint.json` 的主动脉区间（p0.5=122, p99.5=950）比 Merlin 的固定 [-1000,1000]
   更贴合 aortic root 的心腔对比；**A 组用 fingerprint，B 组用 Merlin 原生 [-1000,1000]**，这是刻意的对照。
3. 钙化任务保留 `clip [-100, 2000]`（钙化前景 p0.5=538 / p99.5=2032），因为 Merlin 的 [-1000,1000]
   会把钙化饱和掉 —— 这一项必须单独开实验确认（见 E7）。

---

## 3. 模型设计

```
Merlin I3ResNet (return_skips=True, BN frozen)
  skip@conv1: 64ch  (1/4)          ← 若是 B 方案，1/4 = 56×56×40
  skip@layer1: 256ch (1/4)
  skip@layer2: 512ch (1/8)
  skip@layer3: 1024ch (1/16)
  skip@layer4: 2048ch (1/32)
        ↓
UNet 解码器 (2× 上采样 + 3×3×3 conv + InstanceNorm + ReLU, 通道 256→128→64→32)
        ↓
任务头：
  - landmark: 5ch (sigmoid 在外部)      [与基线热图定义一致]
  - plane:    6ch = 5 landmark + ring   [与 plane 基线一致]
  - calc:     1ch (logits)              [与 calcification 基线一致]
```

- 解码器规模：~12–15 M 参数，占可训练参数的 20%。
- 三任务**分别训练**（先 landmark，再 plane 从 landmark 初始化，最后 calc），与现有流程一致，便于控制变量。
  多任务联合训练留作 E8。

---

## 4. 实验矩阵

| ID | Backbone | 初始化 | 输入 | 目标任务 | 目的 |
|---|---|---|---|---|---|
| E0 | SegResNetDS | scratch | 0.5/192³ | landmark | **基线复现**（必须能跑出现有 MRE） |
| E1 | Merlin + dec | 冻结 encoder（linear probe 变体：只训解码器） | 0.5/192³ | landmark | 测表示质量 |
| E2 | Merlin + dec | 全量微调（encoder lr 1e-5） | 0.5/192³ | landmark | 主实验 A |
| E2b | Merlin + dec | 全量微调（encoder lr 5e-5） | 0.5/192³ | landmark | lr 敏感性 |
| E3 | Merlin + dec | 全量微调 | 1.5×1.5×3 / 224×224×160 | landmark | 主实验 B |
| E4 | E2 最佳 | 传到 plane | 0.5/192³ | plane (6ch) | 平面任务 |
| E5 | E2 最佳 | 传到 calc | 0.5/192³ | calcification | 钙化任务 |
| E6 | E2 最佳 | — | — | landmark | **FOV 消融**：96×96×72 mm vs 144×144×144 mm |
| E7 | E5 | — | 0.5/192³ | calcification | 归一化消融：fingerprint vs [-1000,1000] |
| E8 | 同上 | — | — | 三任务联合 | 只做一次，看是否互相干扰 |
| E9 | Merlin | 领域继续预训练（2790 例）后再 E2 | — | landmark | 看 pretrain 集合值不值得 |

优先级：E0 → E1/E2 → E3 → E4/E5 → 其余按结果定。

**统一评估协议**（沿用管线，不改口径）：

- 3-fold CV（`splits.json` 的 huaxi_dev 151）+ `huaxi_test` 38 + `carry_external` 223 外部验证
- Landmark：逐点 **MRE (mm)**、**SDR@2/5/10 mm**；缺失 GT 的点不参与统计（BAV 病例 / carry 缺 LCA/RCA）
- Plane：环拟合平面法向夹角、瓣环中心距离
- Calcification：Dice / HD95 + 连通域级检出率
- 同一 fold 的 train/val 划分、同一 seed，报告 mean ± std over 3 folds

---

## 5. 训练配置

```yaml
optimizer: AdamW
lr:  head 2e-4 / decoder 2e-4 / encoder 1e-5 (E2b: 5e-5, E1: frozen)
scheduler: cosine + 10 epoch warmup
weight_decay: 1e-5
amp: bf16 (A100) 或 fp16 (其他)
batch: 2–4（按显存；不足用 grad accumulation 保持有效 batch 4）
epochs: 300 上限, early stopping patience 50 (val MRE)
aug: 沿用现有（rotate ±0.15rad, translate ±5 voxel, scale ±10%, 高斯噪声 0.05, 空间增强 p=0.3）
     ⚠ 禁止 flip（L/R、LCA/RCA 有解剖语义）
BN: encoder 全程 eval + 冻结（关键）
```

预期显存（fp16, 192×192×144, batch 2）：**~18–24 GB**；A 方案下不需要 gradient checkpointing
（Merlin 各 layer 内部已有 checkpoint），B 方案（224×224×160）建议 batch 1–2。

粗略算力：单样本前向 ~4 TFLOPs，训练 ~12 TFLOPs；
151 例 × 300 ep × 3 fold ≈ **A100 上 3–4 GPU·h/任务，4090 上 8–10 GPU·h/任务**。

---

## 6. 排期（4 周）

| 周 | 交付 |
|---|---|
| W1 | 服务器环境（conda + monai + merlin-vlm）、只读数据软链/复制、`MerlinDataset`、**E0 基线复现跑通并复现现有指标** |
| W2 | E1 / E2 / E2b（landmark 3-fold），出 MRE/SDR 对照表 |
| W3 | E3（原生 spacing）、E4（plane）、E5（calcification） |
| W4 | E6/E7 消融 + 汇总图 + 结论（Merlin 是否值得作为默认 backbone） |

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 0.5 mm 输入与 Merlin 预训练 spacing 差 3× | 迁移收益变负 | 这就是 E3 存在的意义；若 E3 明显更好但精度不足，转**级联**（Merlin@1.5 mm 粗定位 → 0.5 mm 细化） |
| BN 在 batch 1–4 下统计量崩 | 训练发散 | 冻结 BN；若仍不稳，用 GN 重建 BN（会丢预训练统计，需重训几 epoch） |
| 3 mm 层厚下 landmark 精度天花板 | SDR@2 mm 不合格 | 峰值用**亚体素质心**（不是 argmax）；只在 E3 出问题时才考虑 |
| 420 例下游样本量小 | 过拟合 | 3-fold + 早停；E9 领域继续预训练 |
| 旧管线目录只读 | 无法写入新产物 | 预训练/新产物一律写 `~/program/CTA-Foudation-model/work/`，只读引用旧 crop |
| 磁盘 | huaxi+carry 全量 nii 约 400×200 MB ≈ 80 GB | 先确认 `/media/dell` 剩余空间；预训练 2790 例按需流式读取，不落缓存 |

---

## 8. 待你确认

1. **服务器 GPU 型号 / 显存 / 张数**（决定 E2 的 batch 和是否走 B 方案）。
2. 旧 `/data/disk_c/valve_pipeline` 的 crop/热图能否直接只读复用（现在对 `hx-dell-dev` 是 `r-x`），
   还是必须在 release_v1 上重跑 step5–step6（约需 TotalSegmentator GPU 跑 420+2790 例）。
3. 验收口径：landmark MRE 目标 / SDR@2 mm 目标是否有硬指标？
4. 是否需要把 2790 例 pretrain 纳入路线（E9），还是只做下游微调。

---

## 附：环境（服务器侧建议）

```bash
conda create -n merlin-valve python=3.10 -y && conda activate merlin-valve
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install monai SimpleITK nibabel PyYAML tensorboard tqdm
pip install -e ./Merlin          # 或 pip install merlin-vlm
# 权重：Merlin/merlin/models/checkpoints/i3_resnet_clinical_longformer_best_clip_*.pt (1.08 GB)
# 文本分支的两个 HF 依赖（Clinical-Longformer、resnet152）跑图像-only 时也需要下载
```

国内网络注意：`huggingface.co` 直连不通，`hf-mirror.com` 时好时坏；
本机实测可用 Clash 代理 `127.0.0.1:7897`（`export HTTPS_PROXY=http://127.0.0.1:7897`）。
