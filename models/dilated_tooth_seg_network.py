import lightning as L
import torch.nn.functional as F
import torch
import torchmetrics as tm
import numpy as np
from torch import nn
from models.layer import (
    BasicPointLayer,
    EdgeGraphConvBlock,
    DilatedEdgeGraphConvBlock,
    ResidualBasicPointLayer,
    PointFeatureImportance,
    STNkd,
)
from libs.pointops.functions import pointops


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.clone())
        else:
            self.weight = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C, N], targets: [B, N]
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=getattr(self, "weight", None),
            reduction="none",
        )
        probs = torch.softmax(logits, dim=1)
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        modulating = (1.0 - target_probs).clamp(min=1e-6) ** self.gamma
        loss = modulating * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def prepare_pointops_format(positions, batch_size):
    """
    将 [B, N, 3] 格式的位置数据转换为 pointops 需要的格式
    """
    B, N, _ = positions.shape
    # 重塑为 [B*N, 3]
    positions_flat = positions.reshape(-1, 3).contiguous()
    # 创建 offset: [0, N, 2*N, ..., B*N]
    offset = torch.arange(0, (B+1)*N, N, dtype=torch.int32, device=positions.device)
    return positions_flat, offset


class BoundaryContrastiveLoss(nn.Module):
    def __init__(self, nsample=12, temperature=0.07):
        super().__init__()
        self.nsample = nsample
        self.temperature = temperature

    def forward(self, features, positions, labels):
        """
        Args:
            features: [B, N, C] 特征
            positions: [B, N, 3] 位置
            labels: [B, N] 标签
        """
        B, N, C = features.shape
        # 准备 pointops 格式
        positions_flat, offset = prepare_pointops_format(positions, B)
        labels_flat = labels.reshape(-1)  # [B*N]
        features_flat = features.reshape(-1, C)  # [B*N, C]

        # 一次性检测所有边界点
        boundary_mask = self._detect_boundary_points_vectorized(
            labels_flat, positions_flat, offset, B, N, k=self.nsample
        )

        if not boundary_mask.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 获取边界点
        boundary_indices = torch.where(boundary_mask)[0]
        boundary_features = features_flat[boundary_indices]
        boundary_labels = labels_flat[boundary_indices]
        boundary_positions = positions_flat[boundary_indices]

        # 计算边界点间的对比损失
        loss = self._compute_contrastive_loss_vectorized(
            boundary_features, boundary_labels, boundary_positions
        )

        return loss

    def _detect_boundary_points_vectorized(self, labels_flat, positions_flat, offset, B, N, k=12):
        """使用 pointops 向量化检测边界点并进行邻域平滑"""
        total_points = B * N

        neighbor_idx, _ = pointops.knnquery(k + 1, positions_flat, positions_flat, offset, offset)
        neighbor_idx = neighbor_idx[:, 1:]

        neighbor_labels = labels_flat[neighbor_idx]
        current_labels = labels_flat.unsqueeze(1)

        diff_mask = neighbor_labels != current_labels
        different_ratio = diff_mask.float().mean(dim=1)

        # 几何相似度平滑：考虑邻居的差异比率，抑制孤立噪声
        neighbor_ratio = different_ratio[neighbor_idx]
        smoothed_ratio = 0.7 * different_ratio + 0.3 * neighbor_ratio.mean(dim=1)

        boundary_mask = smoothed_ratio > 0.5

        return boundary_mask

    def _compute_contrastive_loss_vectorized(self, boundary_features, boundary_labels, boundary_positions):
        """向量化计算对比损失"""
        M = boundary_features.shape[0]
        if M < 2:
            return torch.tensor(0.0, device=boundary_features.device, requires_grad=True)

        # 准备单批次 offset
        offset_single = torch.tensor([0, M], dtype=torch.int32, device=boundary_positions.device)

        # 获取边界点之间的邻居
        neighbor_idx, _ = pointops.knnquery(
            min(self.nsample, M-1), boundary_positions, boundary_positions,
            offset_single, offset_single
        )

        # 特征归一化
        boundary_features = F.normalize(boundary_features, dim=-1)

        # 向量化计算相似度
        anchor_features = boundary_features.unsqueeze(1)  # [M, 1, C]
        neighbor_features = boundary_features[neighbor_idx]  # [M, K, C]

        # 计算相似度矩阵
        sim_matrix = torch.sum(anchor_features * neighbor_features, dim=-1) / self.temperature  # [M, K]

        # 计算正样本mask
        anchor_labels = boundary_labels.unsqueeze(1)  # [M, 1]
        neighbor_labels = boundary_labels[neighbor_idx]  # [M, K]
        pos_mask = (anchor_labels == neighbor_labels)  # [M, K]

        # 过滤掉没有正负样本对比的点
        has_pos = pos_mask.any(dim=1)
        has_neg = (~pos_mask).any(dim=1)
        valid_mask = has_pos & has_neg

        if not valid_mask.any():
            return torch.tensor(0.0, device=boundary_features.device, requires_grad=True)

        # 只计算有效点的损失
        sim_matrix = sim_matrix[valid_mask]  # [V, K]
        pos_mask = pos_mask[valid_mask]      # [V, K]

        # 计算 InfoNCE 损失
        exp_sim = torch.exp(sim_matrix)
        pos_exp = (exp_sim * pos_mask.float()).sum(dim=1)
        all_exp = exp_sim.sum(dim=1)

        loss = -torch.log(pos_exp / all_exp + 1e-8)
        return loss.mean()


# Bmiou
class BoundaryMIoU:
    @staticmethod
    def compute_boundary_miou(pred_labels, true_labels, positions, k=12):
        """
        计算边界点的MIoU
        Args:
            pred_labels: [B, N] 预测标签
            true_labels: [B, N] 真实标签
            positions: [B, N, 3] 位置信息
        """
        B, N = pred_labels.shape

        # 准备 pointops 格式
        positions_flat, offset = prepare_pointops_format(positions, B)
        pred_labels_flat = pred_labels.reshape(-1)
        true_labels_flat = true_labels.reshape(-1)

        # 向量化检测边界点
        boundary_mask = BoundaryMIoU._detect_boundary_points_vectorized(
            true_labels_flat, positions_flat, offset, k
        )

        if not boundary_mask.any():
            return torch.tensor(0.0, device=pred_labels.device)

        # 获取边界点的标签
        boundary_pred = pred_labels_flat[boundary_mask]
        boundary_true = true_labels_flat[boundary_mask]

        # 计算边界点的类别范围
        unique_true_labels = torch.unique(boundary_true)

        if len(unique_true_labels) <= 1:
            return torch.tensor(0.0, device=pred_labels.device)

        # 计算MIoU
        return BoundaryMIoU._compute_miou_for_classes(
            boundary_pred, boundary_true, unique_true_labels
        )

    @staticmethod
    def _detect_boundary_points_vectorized(labels_flat, positions_flat, offset, k=12):
        neighbor_idx, _ = pointops.knnquery(k + 1, positions_flat, positions_flat, offset, offset)
        neighbor_idx = neighbor_idx[:, 1:]

        neighbor_labels = labels_flat[neighbor_idx]
        current_labels = labels_flat.unsqueeze(1)
        diff_mask = neighbor_labels != current_labels
        different_ratio = diff_mask.float().mean(dim=1)
        neighbor_ratio = different_ratio[neighbor_idx]
        smoothed_ratio = 0.7 * different_ratio + 0.3 * neighbor_ratio.mean(dim=1)

        return smoothed_ratio > 0.45

    @staticmethod
    def _compute_miou_for_classes(pred_labels, true_labels, class_labels):
        """计算指定类别的MIoU"""
        ious = []

        for cls in class_labels:
            pred_mask = (pred_labels == cls)
            true_mask = (true_labels == cls)

            intersection = (pred_mask & true_mask).sum().float()
            union = (pred_mask | true_mask).sum().float()

            if union > 0:
                iou = intersection / union
                ious.append(iou)

        return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=pred_labels.device)


class BoundaryAwareMultiScaleFusion(nn.Module):
    """改进版边界感知多尺度融合模块"""

    def __init__(
        self,
        feat_dims,
        n_scales: int = 3,
        reduce_dim: int = 320,
        boundary_feat_dim: int = 6,
        logit_temperature: float = 0.75,
        boundary_k: int = 12,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.logit_temperature = logit_temperature
        self.boundary_k = boundary_k

        self.feat_projs = nn.ModuleList([nn.Linear(dim, reduce_dim) for dim in feat_dims])

        self.boundary_encoder = nn.Sequential(
            nn.Linear(boundary_feat_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 160),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(reduce_dim + 160, 320),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(320, n_scales),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(reduce_dim, reduce_dim),
            nn.ReLU(),
            nn.Linear(reduce_dim, reduce_dim),
        )

    def extract_boundary_info(self, logits, labels, pos, use_gt: bool = True):
        if logits.dim() == 3 and logits.shape[1] != pos.shape[1]:
            logits = logits.transpose(1, 2)

        B, N = (labels.shape if use_gt else logits.shape[:2])
        target_labels = labels if use_gt else torch.argmax(logits, dim=-1)

        positions_flat, offset = prepare_pointops_format(pos, B)
        labels_flat = target_labels.reshape(-1)

        neighbor_idx, _ = pointops.knnquery(
            self.boundary_k + 1, positions_flat, positions_flat, offset, offset
        )
        neighbor_idx = neighbor_idx[:, 1:]

        neighbor_labels = labels_flat[neighbor_idx]
        current_labels = labels_flat.unsqueeze(1)
        diff_mask = neighbor_labels != current_labels
        different_ratio = diff_mask.float().mean(dim=1)

        neighbor_positions = positions_flat[neighbor_idx]
        center_positions = positions_flat.unsqueeze(1)
        neighbor_vec = neighbor_positions - center_positions
        distances = torch.linalg.norm(neighbor_vec, dim=-1)

        same_label_mask = (~diff_mask).float()
        same_label_dist = (
            (distances * same_label_mask).sum(dim=1)
            / (same_label_mask.sum(dim=1) + 1e-6)
        )
        boundary_distance = (
            torch.where(diff_mask, distances, torch.full_like(distances, float("inf"))).min(dim=1).values
        )
        boundary_distance = torch.where(
            torch.isfinite(boundary_distance), boundary_distance, same_label_dist
        )

        density = 1.0 / (distances.mean(dim=1) + 1e-6)
        curvature = distances.std(dim=1) / (distances.mean(dim=1) + 1e-6)

        logits_scaled = logits / self.logit_temperature
        probs = F.softmax(logits_scaled, dim=-1)
        confidence = probs.max(dim=-1)[0]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) / np.log(probs.shape[-1])

        boundary_score = different_ratio.reshape(B, N)
        density = density.reshape(B, N)
        curvature = curvature.reshape(B, N)
        boundary_distance = boundary_distance.reshape(B, N)

        return torch.stack(
            [boundary_score, confidence, entropy, density, curvature, boundary_distance],
            dim=-1,
        )

    def forward(self, feats, logits, labels, pos):
        B, N = pos.shape[:2]

        if logits.dim() == 3 and logits.shape[1] != N:
            logits = logits.transpose(1, 2)

        feats_proj = [proj(f) for proj, f in zip(self.feat_projs, feats)]
        feats_stack = torch.stack(feats_proj, dim=2)

        use_gt = (labels is not None) and self.training
        boundary_info = self.extract_boundary_info(
            logits, labels if use_gt else None, pos, use_gt=use_gt
        )

        boundary_encoding = self.boundary_encoder(boundary_info)

        global_feat = feats_stack.mean(dim=2)
        attn_input = torch.cat([global_feat, boundary_encoding], dim=-1)
        attn_weights = F.softmax(self.attention(attn_input), dim=-1)

        fused_feat = (feats_stack * attn_weights.unsqueeze(-1)).sum(dim=2)
        output = self.output_proj(fused_feat) + global_feat

        return output, attn_weights


class DilatedToothSegmentationNetwork(nn.Module):
    def __init__(self, num_classes=17, feature_dim=24):
        """
        :param num_classes: Number of classes to predict
        """
        super(DilatedToothSegmentationNetwork, self).__init__()
        self.num_classes = num_classes

        self.stnkd = STNkd(k=24)

        self.edge_graph_conv_block1 = EdgeGraphConvBlock(in_channels=feature_dim, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")
        self.edge_graph_conv_block2 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")
        self.edge_graph_conv_block3 = EdgeGraphConvBlock(in_channels=24, out_channels=24, k=32,
                                                         hidden_channels=24,
                                                         edge_function="local_global")

        self.local_hidden_layer = BasicPointLayer(in_channels=24 * 3, out_channels=60)

        self.dilated_edge_graph_conv_block1 = DilatedEdgeGraphConvBlock(
            in_channels=60,
            hidden_channels=60,
            out_channels=60,
            k=32,
            dilation_k=200,
            edge_function="local_global",
        )
        self.dilated_edge_graph_conv_block2 = DilatedEdgeGraphConvBlock(
            in_channels=60,
            hidden_channels=60,
            out_channels=60,
            k=32,
            dilation_k=900,
            edge_function="local_global",
        )
        self.dilated_edge_graph_conv_block3 = DilatedEdgeGraphConvBlock(
            in_channels=60,
            hidden_channels=60,
            out_channels=60,
            k=32,
            dilation_k=1800,
            edge_function="local_global",
        )
        self.dilated_edge_graph_conv_block4 = DilatedEdgeGraphConvBlock(
            in_channels=60,
            hidden_channels=60,
            out_channels=60,
            k=32,
            dilation_k=2400,
            edge_function="local_global",
        )

        self.bamsf = BoundaryAwareMultiScaleFusion(
            feat_dims=[72, 60, 240],
            n_scales=3,
            reduce_dim=320,
        )

        self.temp_classifier = nn.Sequential(
            nn.Linear(300, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(160, num_classes),
        )

        self.local_aux_head = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.mid_aux_head = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.global_aux_head = nn.Sequential(
            nn.Linear(240, 160),
            nn.ReLU(),
            nn.Linear(160, num_classes),
        )
        self.fused_aux_head = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Linear(160, num_classes),
        )

        self.feature_importance = PointFeatureImportance(in_channels=320)
        self.res_block1 = ResidualBasicPointLayer(
            in_channels=320, out_channels=448, hidden_channels=448
        )
        self.res_block2 = ResidualBasicPointLayer(
            in_channels=448, out_channels=320, hidden_channels=320
        )
        self.out = BasicPointLayer(in_channels=320, out_channels=num_classes, is_out=True)

        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x, pos, labels=None):
        # precompute pairwise distance of points
        cd = torch.cdist(pos, pos)
        x = self.stnkd(x)
        # 局部特征
        x1, _ = self.edge_graph_conv_block1(x, pos)
        x2, _ = self.edge_graph_conv_block2(x1)
        x3, _ = self.edge_graph_conv_block3(x2)

        x_local = torch.cat([x1, x2, x3], dim=2)

        x_mid = self.local_hidden_layer(x_local)

        x_d1, _ = self.dilated_edge_graph_conv_block1(x_mid, pos, cd=cd)
        x_d2, _ = self.dilated_edge_graph_conv_block2(x_d1, pos, cd=cd)
        x_d3, _ = self.dilated_edge_graph_conv_block3(x_d2, pos, cd=cd)
        x_d4, _ = self.dilated_edge_graph_conv_block4(x_d3, pos, cd=cd)
        x_global = torch.cat([x_d1, x_d2, x_d3, x_d4], dim=2)

        aux_logits_local = self.local_aux_head(x_local)
        aux_logits_mid = self.mid_aux_head(x_mid)
        aux_logits_global = self.global_aux_head(x_global)

        x_temp = torch.cat([x_mid, x_d1, x_d2, x_d3, x_d4], dim=2)  # [B, N, 300]
        logits_temp = self.temp_classifier(x_temp)

        feats = [x_local, x_mid, x_global]
        x_fused, attn_weights = self.bamsf(feats, logits_temp, labels, pos)
        logits_fused = self.fused_aux_head(x_fused)
        x_fused = self.dropout2(x_fused)

        x = self.feature_importance(x_fused)
        x = self.res_block1(x)
        features = self.res_block2(x)
        features = self.dropout3(features)
        seg_pred = self.out(features)

        aux_logits = {
            "local": aux_logits_local,
            "mid": aux_logits_mid,
            "global": aux_logits_global,
            "temp": logits_temp,
            "fused": logits_fused,
        }

        return seg_pred, features, x_fused, aux_logits



class LitDilatedToothSegmentationNetwork(L.LightningModule):
    def __init__(
        self,
        boundary_contrast_weight: float = 1.2,
        boundary_warmup_epochs: int = 10,
        aux_local_weight: float = 0.2,
        aux_mid_weight: float = 0.2,
        aux_global_weight: float = 0.25,
        aux_temp_weight: float = 0.15,
        aux_fused_weight: float = 0.3,
        class_weights=None,
        use_focal_loss: bool = False,
        focal_gamma: float = 1.5,
        boundary_contrast_nsample: int = 12,
        boundary_contrast_temperature: float = 0.07,
        bmiou_k: int = 12,
        lr: float = 1e-3,
        lr_min: float = 1e-5,
        lr_restart_interval: int = 50,
        lr_restart_mult: int = 2,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = DilatedToothSegmentationNetwork(num_classes=17, feature_dim=24)
        self.bmiou_k = bmiou_k

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float)
            self.register_buffer('class_weight_tensor', weight_tensor)
            weight_for_losses = self.class_weight_tensor
        else:
            self.class_weight_tensor = None
            weight_for_losses = None

        if use_focal_loss:
            self.seg_loss = FocalLoss(weight=weight_for_losses, gamma=focal_gamma)
        else:
            self.seg_loss = nn.CrossEntropyLoss(weight=weight_for_losses)
        self.aux_loss = nn.CrossEntropyLoss(weight=weight_for_losses)

        self.boundary_contrast_loss = BoundaryContrastiveLoss(
            nsample=boundary_contrast_nsample, temperature=boundary_contrast_temperature
        )
        self.boundary_contrast_weight_max = boundary_contrast_weight
        self.boundary_warmup_epochs = max(1, boundary_warmup_epochs)
        self.boundary_weight = 0.0
        self.boundary_loss_enabled = False

        self.aux_weights = {
            'local': aux_local_weight,
            'mid': aux_mid_weight,
            'global': aux_global_weight,
            'temp': aux_temp_weight,
            'fused': aux_fused_weight,
        }

        self.lr = lr
        self.lr_min = lr_min
        self.lr_restart_interval = lr_restart_interval
        self.lr_restart_mult = lr_restart_mult
        self.weight_decay = weight_decay

        self.train_acc = tm.Accuracy(task='multiclass', num_classes=17)
        self.val_acc = tm.Accuracy(task='multiclass', num_classes=17)
        self.test_acc = tm.Accuracy(task='multiclass', num_classes=17)
        self.train_miou = tm.JaccardIndex(task='multiclass', num_classes=17)
        self.val_miou = tm.JaccardIndex(task='multiclass', num_classes=17)
        self.test_miou = tm.JaccardIndex(task='multiclass', num_classes=17)

        self.best_val_miou = 0.0

        self.save_hyperparameters({
            'boundary_contrast_weight': boundary_contrast_weight,
            'boundary_warmup_epochs': boundary_warmup_epochs,
            'aux_local_weight': aux_local_weight,
            'aux_mid_weight': aux_mid_weight,
            'aux_global_weight': aux_global_weight,
            'aux_temp_weight': aux_temp_weight,
            'aux_fused_weight': aux_fused_weight,
            'use_focal_loss': use_focal_loss,
            'focal_gamma': focal_gamma,
            'boundary_contrast_nsample': boundary_contrast_nsample,
            'boundary_contrast_temperature': boundary_contrast_temperature,
            'bmiou_k': bmiou_k,
            'lr': lr,
            'lr_min': lr_min,
            'lr_restart_interval': lr_restart_interval,
            'lr_restart_mult': lr_restart_mult,
            'weight_decay': weight_decay,
        })

    def _current_boundary_weight(self) -> float:
        progress = min(1.0, (self.current_epoch + 1) / self.boundary_warmup_epochs)
        return self.boundary_contrast_weight_max * progress

    def _compute_aux_losses(self, aux_logits, targets):
        aux_losses = {}
        total = 0.0
        for name, weight in self.aux_weights.items():
            if weight <= 0:
                continue
            logits = aux_logits[name].transpose(2, 1)
            loss = self.aux_loss(logits, targets)
            aux_losses[name] = loss
            total = total + weight * loss
        return total, aux_losses

    def training_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, _ = x.shape
        x = x.float()
        y = y.reshape(B, N).long()

        seg_pred, features, x_fused, aux_logits = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)

        seg_loss = self.seg_loss(seg_pred, y)

        boundary_loss1 = self.boundary_contrast_loss(x_fused, pos, y)
        boundary_loss2 = self.boundary_contrast_loss(features, pos, y)
        boundary_loss = 0.5 * (boundary_loss1 + boundary_loss2)
        boundary_term = self.boundary_weight * boundary_loss

        aux_total_loss, aux_losses = self._compute_aux_losses(aux_logits, y)

        total_loss = seg_loss + boundary_term + aux_total_loss

        self.train_acc(seg_pred, y)
        self.train_miou(seg_pred, y)
        # 计算BMIoU
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos, k=self.bmiou_k)

        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_miou', self.train_miou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_bmiou', bmiou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_boundary_loss', boundary_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_boundary_weight', torch.tensor(self.boundary_weight, device=seg_loss.device),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_aux_total', aux_total_loss, on_step=False, on_epoch=True, sync_dist=True)
        for name, loss in aux_losses.items():
            self.log(f'train_aux_{name}', loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, _ = x.shape
        x = x.float()
        y = y.reshape(B, N).long()

        seg_pred, features, x_fused, aux_logits = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)

        seg_loss = self.seg_loss(seg_pred, y)
        boundary_loss1 = self.boundary_contrast_loss(x_fused, pos, y)
        boundary_loss2 = self.boundary_contrast_loss(features, pos, y)
        boundary_loss = 0.5 * (boundary_loss1 + boundary_loss2)
        aux_total_loss, aux_losses = self._compute_aux_losses(aux_logits, y)
        total_loss = seg_loss + self.boundary_weight * boundary_loss + aux_total_loss

        self.val_acc(seg_pred, y)
        self.val_miou(seg_pred, y)
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos, k=self.bmiou_k)

        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_miou', self.val_miou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_bmiou', bmiou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_seg_loss', seg_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_boundary_loss', boundary_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_aux_total', aux_total_loss, on_step=False, on_epoch=True, sync_dist=True)
        for name, loss in aux_losses.items():
            self.log(f'val_aux_{name}', loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return

        current_val_miou = float(self.trainer.logged_metrics.get('val_miou', 0.0))
        if current_val_miou > self.best_val_miou:
            self.best_val_miou = current_val_miou

    def test_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, _ = x.shape
        x = x.float()
        y = y.reshape(B, N).long()

        seg_pred, features, x_fused, aux_logits = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)

        seg_loss = self.seg_loss(seg_pred, y)
        boundary_loss1 = self.boundary_contrast_loss(x_fused, pos, y)
        boundary_loss2 = self.boundary_contrast_loss(features, pos, y)
        boundary_loss = 0.5 * (boundary_loss1 + boundary_loss2)
        aux_total_loss, aux_losses = self._compute_aux_losses(aux_logits, y)
        total_loss = seg_loss + self.boundary_contrast_weight_max * boundary_loss + aux_total_loss

        # 主模型指标
        self.test_acc(seg_pred, y)
        self.test_miou(seg_pred, y)

        # 主模型BMIoU
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos, k=self.bmiou_k)

        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_miou', self.test_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_bmiou', bmiou, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_boundary_loss', boundary_loss, on_step=False, on_epoch=True)
        self.log('test_aux_total', aux_total_loss, on_step=False, on_epoch=True)
        for name, loss in aux_losses.items():
            self.log(f'test_aux_{name}', loss, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_start(self):
        self.boundary_weight = self._current_boundary_weight()
        self.boundary_loss_enabled = self.boundary_weight > 1e-8
        self.train_acc.reset()
        self.train_miou.reset()

    def on_validation_epoch_start(self):
        """验证epoch开始时重置指标"""
        self.val_acc.reset()
        self.val_miou.reset()

    def predict_labels(self, data):
        with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            with torch.no_grad():
                pos, x, y = data
                pos = pos.unsqueeze(0).to(self.device)
                x = x.unsqueeze(0).to(self.device)
                B, N, _ = x.shape
                x = x.float()

                seg_pred, _, _, _ = self.model(x, pos, labels=y)
                pred_labels = torch.argmax(seg_pred, dim=1)

                return pred_labels.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.lr_restart_interval,
            T_mult=self.lr_restart_mult,
            eta_min=self.lr_min,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
            },
        }
