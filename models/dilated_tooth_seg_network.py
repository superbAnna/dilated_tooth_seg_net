import lightning as L
import torch.nn.functional as F
import torch
import torchmetrics as tm
import numpy as np
from torch import nn
from models.layer import BasicPointLayer, EdgeGraphConvBlock, DilatedEdgeGraphConvBlock, ResidualBasicPointLayer, \
    PointFeatureImportance, STNkd
from libs.pointops.functions import pointops


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
    def __init__(self, nsample=8, temperature=0.07):
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
            labels_flat, positions_flat, offset, B, N
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

    def _detect_boundary_points_vectorized(self, labels_flat, positions_flat, offset, B, N, k=8):
        """使用 pointops 向量化检测边界点"""
        total_points = B * N

        # 使用 pointops 进行 KNN 查询
        neighbor_idx, _ = pointops.knnquery(k+1, positions_flat, positions_flat, offset, offset)
        neighbor_idx = neighbor_idx[:, 1:]  # 排除自己 [total_points, k]

        # 向量化计算边界点
        # 获取邻居标签
        neighbor_labels = labels_flat[neighbor_idx]  # [total_points, k]
        current_labels = labels_flat.unsqueeze(1)    # [total_points, 1]

        # 计算不同标签的数量
        different_count = (neighbor_labels != current_labels).sum(dim=1)  # [total_points]

        # 边界点判断：超过一半邻居标签不同
        boundary_mask = different_count > (k / 2)

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
    def compute_boundary_miou(pred_labels, true_labels, positions, k=8):
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
    def _detect_boundary_points_vectorized(labels_flat, positions_flat, offset, k=8):
        """使用 pointops 向量化检测边界点"""
        # 使用 pointops 进行 KNN 查询
        neighbor_idx, _ = pointops.knnquery(k+1, positions_flat, positions_flat, offset, offset)
        neighbor_idx = neighbor_idx[:, 1:]  # 排除自己

        # 向量化边界检测
        neighbor_labels = labels_flat[neighbor_idx]
        current_labels = labels_flat.unsqueeze(1)
        different_count = (neighbor_labels != current_labels).sum(dim=1)

        return different_count > (k / 2)

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
    """
    改进版边界感知多尺度融合模块
    """
    def __init__(self, feat_dims, n_scales=3, reduce_dim=256):
        """
        Args:
            feat_dims: list of int, 各尺度特征维度 [C_local, C_mid, C_global]
            n_scales: 尺度数量
            reduce_dim: 统一特征维度
        """
        super().__init__()
        self.n_scales = n_scales

        # 特征维度对齐
        self.feat_projs = nn.ModuleList([
            nn.Linear(dim, reduce_dim) for dim in feat_dims
        ])

        # 边界特征提取
        self.boundary_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 3个边界特征
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # 注意力权重生成（改进版）
        self.attention = nn.Sequential(
            nn.Linear(reduce_dim + 128, 256),  # 特征 + 边界编码
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_scales)
        )

        # 最终融合
        self.output_proj = nn.Sequential(
            nn.Linear(reduce_dim, reduce_dim),
            nn.ReLU(),
            nn.Linear(reduce_dim, reduce_dim)
        )

    def extract_boundary_info(self, logits, labels, pos, use_gt=True):
        """提取边界信息"""
        B, N = labels.shape if use_gt else logits.shape[:2]

        # 使用GT或预测标签
        target_labels = labels if use_gt else torch.argmax(logits, dim=-1)

        positions_flat, offset = prepare_pointops_format(pos, B)
        labels_flat = target_labels.reshape(-1)

        # KNN检测边界
        neighbor_idx, _ = pointops.knnquery(9, positions_flat, positions_flat, offset, offset)
        neighbor_idx = neighbor_idx[:, 1:]

        neighbor_labels = labels_flat[neighbor_idx]
        current_labels = labels_flat.unsqueeze(1)
        different_ratio = (neighbor_labels != current_labels).float().mean(dim=1)
        boundary_score = different_ratio.reshape(B, N)

        # 预测置信度和熵
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) / np.log(probs.shape[-1])

        return torch.stack([boundary_score, confidence, entropy], dim=-1)  # [B, N, 3]

    def forward(self, feats, logits, labels, pos):
        """
        Args:
            feats: list of [B, N, C_i], 多尺度特征
            logits: [B, N, num_classes] 或 [B, C, N]
            labels: [B, N] GT标签（训练）或None（推理）
            pos: [B, N, 3]
        Returns:
            fused_feat: [B, N, reduce_dim]
        """
        B, N = pos.shape[:2]

        # 维度转换
        if logits.dim() == 3 and logits.shape[1] != N:
            logits = logits.transpose(1, 2)  # [B, C, N] -> [B, N, C]

        # 1. 特征投影对齐
        feats_proj = [proj(f) for proj, f in zip(self.feat_projs, feats)]  # 每个 [B, N, reduce_dim]
        feats_stack = torch.stack(feats_proj, dim=2)  # [B, N, n_scales, reduce_dim]

        # 2. 提取边界信息
        use_gt = (labels is not None) and self.training
        boundary_info = self.extract_boundary_info(
            logits, labels if use_gt else None, pos, use_gt=use_gt
        )  # [B, N, 3]

        # 3. 边界特征编码
        boundary_encoding = self.boundary_encoder(boundary_info)  # [B, N, 128]

        # 4. 生成注意力权重
        # 使用平均池化的全局特征 + 边界编码
        global_feat = feats_stack.mean(dim=2)  # [B, N, reduce_dim]
        attn_input = torch.cat([global_feat, boundary_encoding], dim=-1)
        attn_weights = self.attention(attn_input)  # [B, N, n_scales]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 5. 加权融合
        attn_weights_exp = attn_weights.unsqueeze(-1)  # [B, N, n_scales, 1]
        fused_feat = (feats_stack * attn_weights_exp).sum(dim=2)  # [B, N, reduce_dim]

        # 6. 输出投影
        output = self.output_proj(fused_feat) + global_feat

        return output, attn_weights  # 返回权重用于可视化


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

        self.dilated_edge_graph_conv_block1 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=200, edge_function="local_global")
        self.dilated_edge_graph_conv_block2 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=900, edge_function="local_global")
        self.dilated_edge_graph_conv_block3 = DilatedEdgeGraphConvBlock(in_channels=60, hidden_channels=60,
                                                                        out_channels=60, k=32,
                                                                        dilation_k=1800, edge_function="local_global")

        self.bamsf = BoundaryAwareMultiScaleFusion(
            feat_dims=[72, 60, 180],  # x1+x2+x3=72, dilated各60
            n_scales=3,
            reduce_dim=256
        )
        # 临时分类头（用于BAMSF的边界检测）
        self.temp_classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 特征处理和分类头
        self.feature_importance = PointFeatureImportance(in_channels=256)
        self.res_block1 = ResidualBasicPointLayer(
            in_channels=256, out_channels=384, hidden_channels=384
        )
        self.res_block2 = ResidualBasicPointLayer(
            in_channels=384, out_channels=256, hidden_channels=256
        )
        self.out = BasicPointLayer(in_channels=256, out_channels=num_classes, is_out=True)

        self.dropout2 = nn.Dropout(0.15)
        self.dropout3 = nn.Dropout(0.3)

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
        x_global = torch.cat([x_d1, x_d2, x_d3], dim=2)
        x_temp = torch.cat([x_mid, x_d1, x_d2, x_d3], dim=2)  # [B, N, 240]
        logits_temp = self.temp_classifier(x_temp)  # [B, N, num_classes]

        # ===== 5. 边界感知多尺度融合（BAMSF）=====
        feats = [x_local, x_mid, x_global]  # 3个不同尺度的特征
        x_fused, attn_weights = self.bamsf(
            feats, logits_temp, labels, pos
        )  # [B, N, 256], [B, N, 3]
        x_fused = self.dropout2(x_fused)
        # ===== 6. 特征处理和分类 =====
        x = self.feature_importance(x_fused)
        x = self.res_block1(x)
        features = self.res_block2(x)
        features = self.dropout3(features)
        seg_pred = self.out(features)
        return seg_pred, features, x_fused


class LitDilatedToothSegmentationNetwork(L.LightningModule):
    def __init__(self, boundary_contrast_weight=0.8, enable_boundary_loss_threshold=0.60,
                 stability_window=3,
                 stability_tolerance=0.02,
                 max_train_val_gap=0.35):
        super().__init__()
        self.model = DilatedToothSegmentationNetwork(num_classes=17, feature_dim=24)
        # 损失函数
        self.seg_loss = nn.CrossEntropyLoss()
        self.boundary_contrast_loss = BoundaryContrastiveLoss(nsample=8, temperature=0.07)
        self.boundary_contrast_weight = boundary_contrast_weight
        # 边界损失动态激活参数
        self.enable_boundary_loss_threshold = enable_boundary_loss_threshold
        self.boundary_loss_enabled = False
        self.stability_window = stability_window
        self.stability_tolerance = stability_tolerance
        self.max_train_val_gap = max_train_val_gap
        # 记录历史指标
        self.recent_val_mious = []
        self.best_val_miou = 0.0
        # 主要指标
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.train_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.val_miou = tm.JaccardIndex(task="multiclass", num_classes=17)
        self.test_acc = tm.Accuracy(task="multiclass", num_classes=17)
        self.test_miou = tm.JaccardIndex(task="multiclass", num_classes=17)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.reshape(B, N).long()
        # 前向
        seg_pred, features, x_fused = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)
        # 计算损失
        seg_loss = self.seg_loss(seg_pred, y)
        # 动态决定是否使用边界损失
        if self.boundary_loss_enabled:
            boundary_loss1 = self.boundary_contrast_loss(x_fused, pos, y)
            boundary_loss2 = self.boundary_contrast_loss(features, pos, y)
            boundary_loss = 0.5 * (boundary_loss1 + boundary_loss2)

            total_loss = seg_loss + self.boundary_contrast_weight * boundary_loss
        else:
            boundary_loss = torch.tensor(0.0, device=seg_loss.device)
            total_loss = seg_loss

        # 计算指标
        self.train_acc(seg_pred, y)
        self.train_miou(seg_pred, y)
        # 计算BMIoU
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos)
        # 日志记录
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_miou", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_bmiou", bmiou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_seg_loss", seg_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_boundary_loss", boundary_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("boundary_loss_active", float(self.boundary_loss_enabled), on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.reshape(B, N).long()

        # 主模型前向传播
        seg_pred, features, x_fused = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)

        # 主模型损失
        seg_loss = self.seg_loss(seg_pred, y)
        self.log("val_seg_loss", seg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.boundary_loss_enabled:
            boundary_loss1 = self.boundary_contrast_loss(x_fused, pos, y)
            boundary_loss2 = self.boundary_contrast_loss(features, pos, y)
            boundary_loss = 0.5 * (boundary_loss1 + boundary_loss2)

            self.log("val_seg_loss", seg_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        else:
            boundary_loss = torch.tensor(0.0, device=seg_loss.device)

        total_loss = seg_loss
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # 主模型指标
        self.val_acc(seg_pred, y)
        self.val_miou(seg_pred, y)
        # 主模型BMIoU
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos)

        # 主模型日志
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_miou", self.val_miou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_bmiou", bmiou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def on_validation_epoch_end(self):
        """在验证epoch结束时检查是否启用边界损失"""
        if not self.trainer.is_global_zero:
            return
        # 获取当前验证指标
        current_val_miou = self.trainer.logged_metrics.get('val_miou', 0.0)
        current_train_miou = self.trainer.logged_metrics.get('train_miou', 0.0)

        # 更新历史记录
        self.recent_val_mious.append(float(current_val_miou))
        if len(self.recent_val_mious) > self.stability_window:
            self.recent_val_mious.pop(0)

        # 检查是否应该启用边界损失
        if not self.boundary_loss_enabled and len(self.recent_val_mious) >= self.stability_window:
            # 条件1: 验证mIoU达到阈值
            condition1 = current_val_miou >= self.enable_boundary_loss_threshold

            # 条件2: 验证mIoU稳定（标准差小于容忍度）
            miou_std = np.std(self.recent_val_mious)
            condition2 = miou_std < self.stability_tolerance

            if condition1 and condition2:
                self.boundary_loss_enabled = True
                print(f"\n{'='*60}")
                print(f"🎯 Boundary Loss ENABLED at Epoch {self.trainer.current_epoch + 1}")
                print(f"  Val mIoU: {current_val_miou:.4f} (threshold: {self.enable_boundary_loss_threshold})")
                print(f"  Stability: {miou_std:.4f} (tolerance: {self.stability_tolerance})")
                print(f"{'='*60}\n")

        # 更新最佳验证mIoU
        if current_val_miou > self.best_val_miou:
            self.best_val_miou = current_val_miou

    def test_step(self, batch, batch_idx):
        pos, x, y = batch
        B, N, C = x.shape
        x = x.float()
        y = y.reshape(B, N).long()

        # 主模型前向传播
        seg_pred, features, x_fused = self.model(x, pos, labels=y)
        seg_pred = seg_pred.transpose(2, 1)

        # 主模型损失
        seg_loss = self.seg_loss(seg_pred, y)
        boundary_loss = self.boundary_contrast_loss(features, pos, y)
        total_loss = seg_loss + self.boundary_contrast_weight * boundary_loss

        # 主模型指标
        self.test_acc(seg_pred, y)
        self.test_miou(seg_pred, y)

        # 主模型BMIoU
        pred_labels = torch.argmax(seg_pred, dim=1)
        bmiou = BoundaryMIoU.compute_boundary_miou(pred_labels, y, pos)

        # 主模型日志
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_miou", self.test_miou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_bmiou", bmiou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_start(self):
        """训练epoch开始时重置指标"""
        self.train_acc.reset()
        self.train_miou.reset()

    def on_validation_epoch_start(self):
        """验证epoch开始时重置指标"""
        self.val_acc.reset()
        self.val_miou.reset()

    def predict_labels(self, data):
        with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            with torch.no_grad():
                pos, x, y = data
                pos = pos.unsqueeze(0).to(self.device)
                x = x.unsqueeze(0).to(self.device)
                B, N, C = x.shape
                x = x.float()

                seg_pred, _, _ = self.model(x, pos, labels=y)
                pred_labels = torch.argmax(seg_pred, dim=1)

                return pred_labels.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }
