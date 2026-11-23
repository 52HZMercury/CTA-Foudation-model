"""
valve_models.landmark.losses — 标志点热力图回归损失函数

组合损失: L = λ_mse * MSE + λ_dice * SoftDice + λ_wing * AdaptiveWingLoss

所有损失均支持 existence_mask 以处理 BAV 患者缺失标志点的情况:
  - existence_mask: (B, C) tensor, 1=存在 0=缺失
  - 缺失通道的 loss 贡献为 0
  - 仅对存在的通道计算均值

参考文献:
  - Adaptive Wing Loss: Wang et al., ICCV 2019
    "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for heatmap regression

    相比 MSE，AWL 对热力图峰值附近（前景）更敏感，
    对远离峰值的区域（背景）使用较小梯度，避免背景主导训练。

    数学形式:
        对于 y ∈ [0, 1] (GT heatmap):
        - 当 |y - ŷ| < θ:  AWL = ω * ln(1 + |y - ŷ| / ε)^(α-y)
        - 当 |y - ŷ| ≥ θ:  AWL = A * |y - ŷ| - C

    其中 A, C 是保证连续性的常数。

    参数:
        omega: 控制非线性部分的曲率 (默认 14)
        theta: 线性/非线性切换阈值 (默认 0.5)
        epsilon: 防止 log(0) 的平滑常数 (默认 1.0)
        alpha: 控制对前景/背景的敏感度差异 (默认 2.1, 必须 > 1)
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 Adaptive Wing Loss

        参数:
            pred:   (B, C, D, H, W) 预测热力图 (已经过 sigmoid)
            target: (B, C, D, H, W) GT 热力图 [0, 1]
            mask:   (B, C) existence mask, 1=存在 0=缺失

        返回:
            标量 loss
        """
        delta = (target - pred).abs()

        # 计算非线性部分的参数（依赖于 target 值）
        # alpha - target: 前景 (target≈1) 时指数小 → 梯度大；
        #                 背景 (target≈0) 时指数大 → 梯度小
        alpha_t = self.alpha - target

        # 阈值处的 A 和 C（保证连续性和可微性）
        A = (
            self.omega
            * (1.0 / (1.0 + (self.theta / self.epsilon) ** alpha_t))
            * alpha_t
            * ((self.theta / self.epsilon) ** (alpha_t - 1.0))
            * (1.0 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1.0 + (self.theta / self.epsilon) ** alpha_t
        )

        # 分段计算
        small_mask = (delta < self.theta).float()
        loss_small = self.omega * torch.log(
            1.0 + (delta / self.epsilon) ** alpha_t
        )
        loss_large = A * delta - C
        loss = small_mask * loss_small + (1.0 - small_mask) * loss_large

        # 应用 existence mask
        if mask is not None:
            # 三态 mask: +1=存在(真峰target), -1=真实不存在(全零target,约束输出空),
            # 0=缺失/数据问题(跳过)。用 abs() 让 -1 通道也参与且正向惩罚假峰,
            # 且不让 -1 把 loss 变负 / 把 num_valid 计小。
            mask_part = mask.abs()
            # mask: (B, C) → (B, C, 1, 1, 1) 广播
            spatial_dims = len(pred.shape) - 2  # 3 for 3D
            mask_expanded = mask_part
            for _ in range(spatial_dims):
                mask_expanded = mask_expanded.unsqueeze(-1)
            loss = loss * mask_expanded

            # 仅对参与的通道 (+1 和 -1) 求均值
            num_valid = mask_part.sum() * pred.shape[2] * pred.shape[3] * pred.shape[4]
            if num_valid > 0:
                return loss.sum() / num_valid
            else:
                return loss.sum() * 0.0  # 全部缺失，返回 0（保持计算图）
        else:
            return loss.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss — 仅在前景区域计算

    只对 target > fg_threshold 的体素计算 Dice，忽略大量背景。
    背景的 false positive 由 MSE 和 AWL 负责压制。

    参数:
        smooth: 平滑常数，防止分母为 0 (默认 1e-5)
        fg_threshold: 前景阈值，target 值高于此才参与 Dice 计算
    """

    def __init__(self, smooth: float = 1e-5, fg_threshold: float = 0.01):
        super().__init__()
        self.smooth = smooth
        self.fg_threshold = fg_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算前景区域 Soft Dice Loss

        参数:
            pred:   (B, C, D, H, W) 预测热力图 (已经过 sigmoid)
            target: (B, C, D, H, W) GT 热力图 [0, 1]
            mask:   (B, C) existence mask, 1=存在 0=缺失

        返回:
            1 - mean(DicePerChannel) 的标量 loss
        """
        B, C = pred.shape[:2]

        # 前景掩蔽用"本样本所有通道 GT 前景的并集"。
        # 只按自己通道掩蔽时，通道 j 在别的标志点上起的假峰完全不进 Dice；MSE/AWL 都是
        # 全图平均（800 万体素里 800 个），也压不住。实测 E2 的假峰 77.8% 落在同病例
        # 另一个标志点上，并集掩蔽正好惩罚这种跨通道泄漏。
        fg_mask = ((target > self.fg_threshold).float().amax(dim=1, keepdim=True)
                   ).expand_as(target)
        pred_fg = pred * fg_mask
        target_fg = target * fg_mask

        # 展平空间维度: (B, C, N)
        pred_flat = pred_fg.reshape(B, C, -1)
        target_flat = target_fg.reshape(B, C, -1)

        # 逐通道 Dice: (B, C)
        intersection = (pred_flat * target_flat).sum(dim=-1)
        pred_sum = (pred_flat ** 2).sum(dim=-1)
        target_sum = (target_flat ** 2).sum(dim=-1)

        dice_per_channel = (
            (2.0 * intersection + self.smooth)
            / (pred_sum + target_sum + self.smooth)
        )

        # 应用 existence mask
        if mask is not None:
            # Dice 测前景重叠; -1 通道 (真实不存在) target 全零无前景, Dice 对其无意义
            # (压假峰是 MSE+AWL 的职责)。故 -1 不参与 Dice: clamp(min=0) 使 -1→0 跳过,
            # 只对 +1 通道算 Dice。
            mask_pos = mask.clamp(min=0)
            dice_per_channel = dice_per_channel * mask_pos
            num_valid = mask_pos.sum()
            if num_valid > 0:
                dice_loss = 1.0 - dice_per_channel.sum() / num_valid
            else:
                dice_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            dice_loss = 1.0 - dice_per_channel.mean()

        return dice_loss


class MaskedMSELoss(nn.Module):
    """支持 existence mask 的 MSE Loss

    参数:
        reduction: "mean" (默认) 或 "sum"
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 Masked MSE Loss

        参数:
            pred:   (B, C, D, H, W) 预测热力图
            target: (B, C, D, H, W) GT 热力图
            mask:   (B, C) existence mask

        返回:
            标量 loss
        """
        mse = (pred - target) ** 2

        if mask is not None:
            # 三态: +1/-1 均参与 (-1 通道全零 target 压假峰), 0 跳过。用 abs()。
            mask_part = mask.abs()
            spatial_dims = len(pred.shape) - 2
            mask_expanded = mask_part
            for _ in range(spatial_dims):
                mask_expanded = mask_expanded.unsqueeze(-1)
            mse = mse * mask_expanded

            num_valid = mask_part.sum() * pred.shape[2] * pred.shape[3] * pred.shape[4]
            if num_valid > 0:
                return mse.sum() / num_valid
            else:
                return mse.sum() * 0.0
        else:
            return mse.mean()


class LandmarkHeatmapLoss(nn.Module):
    """标志点热力图回归的组合损失

    L_total = λ_mse * MSE + λ_dice * SoftDice + λ_wing * AdaptiveWingLoss

    所有子损失均支持 existence_mask 以处理 BAV 患者。

    参数:
        mse_weight:  MSE 损失权重 (默认 1.0)
        dice_weight: Soft Dice 损失权重 (默认 1.0)
        wing_weight: Adaptive Wing Loss 权重 (默认 0.5)
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        dice_weight: float = 1.0,
        wing_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
        self.wing_weight = wing_weight

        self.mse_loss = MaskedMSELoss()
        self.dice_loss = SoftDiceLoss(smooth=1e-5)
        self.wing_loss = AdaptiveWingLoss(
            omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1
        )

        logger.info(
            f"组合损失权重: MSE={mse_weight}, Dice={dice_weight}, "
            f"AWL={wing_weight}"
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """计算组合损失

        参数:
            pred:           (B, C, D, H, W) 预测热力图 (已经过 sigmoid)
            target:         (B, C, D, H, W) GT 热力图 [0, 1]
            existence_mask: (B, C) 标志点存在 mask, 1=存在 0=缺失

        返回:
            dict: {
                "total": 加权总损失,
                "mse": MSE 损失值,
                "dice": Dice 损失值,
                "wing": AWL 损失值,
            }
        """
        # 确保 mask 是 float tensor
        if existence_mask is not None:
            existence_mask = existence_mask.float()

        # 计算各子损失
        loss_mse = self.mse_loss(pred, target, existence_mask)
        loss_dice = self.dice_loss(pred, target, existence_mask)
        loss_wing = self.wing_loss(pred, target, existence_mask)

        # 加权求和
        total = (
            self.mse_weight * loss_mse
            + self.dice_weight * loss_dice
            + self.wing_weight * loss_wing
        )

        return {
            "total": total,
            "mse": loss_mse.detach(),
            "dice": loss_dice.detach(),
            "wing": loss_wing.detach(),
        }

    def forward_no_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        existence_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """DS 低分辨率层用: 只算 AWL + MSE, 跳过 Dice"""
        if existence_mask is not None:
            existence_mask = existence_mask.float()

        loss_mse = self.mse_loss(pred, target, existence_mask)
        loss_wing = self.wing_loss(pred, target, existence_mask)

        total = self.mse_weight * loss_mse + self.wing_weight * loss_wing

        return {
            "total": total,
            "mse": loss_mse.detach(),
            "dice": torch.tensor(0.0),
            "wing": loss_wing.detach(),
        }
