# Dilated Tooth Segmentation Network: Optimisation Notes

## 网络结构概览
- **输入处理**：通过 `STNkd` 对原始点特征进行空间对齐，随后经三层 `EdgeGraphConvBlock` 获取不同 receptive field 的局部特征，并在 `BasicPointLayer` 中压缩成中尺度表示 `x_mid`。【F:models/dilated_tooth_seg_network.py†L228-L268】
- **多尺度扩张感受野**：四层不同 dilation 的 `DilatedEdgeGraphConvBlock` 依次堆叠，拼接后形成 240 维的全局表征 `x_global`，用于补充长距离上下文。【F:models/dilated_tooth_seg_network.py†L270-L314】
- **边界感知融合 (BAMSF)**：融合模块读取局部/中尺度/全局特征与临时 logits，根据边界比例、置信度、熵、密度、曲率和最近异类距离六种几何线索自适应计算注意力权重，并输出 320 维边界增强特征。【F:models/dilated_tooth_seg_network.py†L152-L222】
- **辅助监督**：局部、中尺度、全局、临时及融合分支均接上轻量分类头并参与总损失，迫使各尺度表征在训练早期即可对语义做出区分。【F:models/dilated_tooth_seg_network.py†L316-L361】【F:models/dilated_tooth_seg_network.py†L488-L520】
- **主干输出**：融合特征经 `PointFeatureImportance` 与两段残差 MLP 精炼后输出最终分割 logits，同时保留中间特征供边界损失使用。【F:models/dilated_tooth_seg_network.py†L323-L361】

## 损失与调度策略
- **主损失**：默认交叉熵，可通过 CLI 切换至支持类权重的 `FocalLoss`，重点优化难分样本。【F:models/dilated_tooth_seg_network.py†L367-L407】
- **边界对比损失**：在边界点对齐同类、分离异类，新增 `eps` 归一化与 NaN 防护，保证在低置信度区也能输出稳定梯度。【F:models/dilated_tooth_seg_network.py†L58-L125】
- **Warm-up 机制**：边界损失权重按照 epoch 线性升温，首轮训练不启用，随后逐步攀升至设定上限，避免早期梯度震荡。【F:models/dilated_tooth_seg_network.py†L409-L426】【F:models/dilated_tooth_seg_network.py†L522-L546】
- **稳定性改进**：BAMSF、辅助头与边界损失均加入 `nan_to_num` / 有限值检查，训练日志能够在发现异常时回落到零损失防止发散。【F:models/dilated_tooth_seg_network.py†L189-L222】【F:models/dilated_tooth_seg_network.py†L430-L486】
- **优化器调度**：使用 AdamW + 余弦热重启，可通过命令行调整最小学习率、重启周期及倍数，以适配不同批量与硬件设置。【F:models/dilated_tooth_seg_network.py†L585-L619】

## 提升 mIoU / bIoU 的进一步建议
1. **调整辅助损失权重**：当训练曲线稳定后逐步调高融合/全局分支权重（例如 0.3→0.5），可以在不牺牲主干稳定性的前提下强化长距离边界一致性。
2. **扩充边界对比邻域**：`--boundary_contrast_nsample` 默认为 12，若显存允许可提高到 16-20，能够让边界点对比覆盖更多同类正样本，常见地能带来 0.5~1.0 个百分点的 bMIoU 提升。
3. **类别重加权**：结合数据统计为交叉熵或 Focal Loss 提供 class weights，可以显著改善小类别在训练集上的 IoU；命令行参数 `--class_weights` 支持直接注入这些权重。
4. **数据采样优化**：若希望进一步提高训练集指标，可在数据层面对边界点或稀有类别做过采样，并配合 `--train_batch_size` 或梯度累积放大有效批量，使辅助头的监督信号更加平滑。
5. **后处理/验证策略**：在验证脚本中复用与训练一致的 `bmiou_k` 与边界掩膜平滑设置，避免评估阶段过度敏感于孤立噪点，从而与训练集表现保持一致。

按照上述方向逐步调参与扩展增广策略，可在保持数值稳定的同时，继续追求更高的训练集 mIoU 与 bMIoU。
