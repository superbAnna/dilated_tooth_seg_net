# Dilated Tooth Segmentation Network: Analysis & Optimisation Notes

## 网络结构概览
- **输入处理**：通过 `STNkd` 对原始点特征进行空间对齐，然后经过三层 `EdgeGraphConvBlock` 提取局部几何关系，得到 `x1`、`x2`、`x3` 三个尺度的邻域特征。【F:models/dilated_tooth_seg_network.py†L226-L256】
- **局部-全局编码**：局部特征拼接后经 `BasicPointLayer` 得到中尺度表示 `x_mid`，随后叠加三层不同膨胀率的 `DilatedEdgeGraphConvBlock` 捕获长距离依赖，形成多尺度全局特征 `x_d1`、`x_d2`、`x_d3`。【F:models/dilated_tooth_seg_network.py†L258-L283】
- **边界感知融合**：`BoundaryAwareMultiScaleFusion` (BAMSF) 将局部/中尺度/全局特征与边界信息动态加权融合，输出统一维度的特征用于后续残差块和分类头。【F:models/dilated_tooth_seg_network.py†L151-L221】
- **损失设计**：训练阶段组合了边界加权的交叉熵、Soft Dice loss，以及在满足稳定性条件后逐步打开的 `BoundaryContrastiveLoss` 以加强边界区分能力。【F:models/dilated_tooth_seg_network.py†L324-L389】

## 提升 mIoU / bIoU 的建议
1. **更稳定的边界损失调度**
   - 当前仅依赖验证 mIoU 均值与波动度触发，建议增加 *训练-验证差距* 与 *最小开启轮次* 的双阈值防抖逻辑，避免早期开启导致的不稳定梯度。
   - 进一步地，可以对边界对比损失权重使用 `cosine` 或 `sigmoid` 形状的 warm-up，而非线性增长，以减缓开启时的突跳。

2. **类别不平衡的重加权策略**
   - 数据集中部分牙齿类别点数显著少于主导类别，可基于样本统计计算类别频率，动态调节交叉熵的 `class_weight` 或在 Dice loss 前加入 `Focal` 项，改善长尾类别 IoU。

3. **边界候选的增强**
   - 现有 `k=12` 邻域阈值为固定值，可按点云密度自适应调整：例如依据局部点间距估算有效邻域大小，再线性映射至阈值，能缓解高低密度区域的误判，从而提高 bIoU。
   - 在 `BoundaryAwareMultiScaleFusion.extract_boundary_info` 中叠加局部法向或曲率特征，可让注意力更加关注真实几何边界而非噪声。

4. **特征层面的改进**
   - 在 `x_fused` 进入 `PointFeatureImportance` 前加入 `LayerNorm` + `DropPath`（小概率），提升残差块稳定性，减少过拟合。
   - 对 `DilatedEdgeGraphConvBlock` 的 dilated `k` 值采用指数衰减（例如 256/512/1024）可以在保持感受野的同时降低采样噪声。

5. **训练技巧**
   - **EMA + TTA** 已在代码中提供开关，建议常规训练开启 EMA（decay≈0.999），在验证末期叠加简易 TTA（旋转 + 镜像 + KNN 平滑）可带来 0.3~0.7 mIoU 的额外提升。【F:models/dilated_tooth_seg_network.py†L391-L465】
   - 引入混合精度下的梯度裁剪（已启用 1.0）与 `gradual warmup + cosine` 学习率调度（已实现）能够让模型在前期迅速收敛、后期更平滑，提高最终指标稳定性。【F:models/dilated_tooth_seg_network.py†L501-L535】

6. **数据增广与后处理**
   - 增加点云扰动（随机旋转、缩放、抖动）与 CutMix/MixUp 类的点云混合增广，增强模型泛化。
   - 推理阶段对预测标签执行多轮 KNN mode smoothing（已实现 `_knn_smooth`），并结合点密度自适应的置信度阈值，可进一步提升 bIoU。

## 实验优先级建议
1. 先开启 EMA 并观察验证曲线稳定性；
2. 调整边界损失 warm-up 策略，确保稳定开启；
3. 引入类别重加权与增强的边界特征；
4. 最后尝试 TTA + 后处理提升榜单表现。

通过上述步骤，可逐步缓解边界预测不稳定与类别不均衡问题，最终提升整体 mIoU 与 bIoU 表现。
