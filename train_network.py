import argparse
import random
import numpy as np
import os
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork
from datetime import timedelta
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)




def get_dataset(train_test_split=1) -> Dataset:

    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed_torch',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)
    train = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed_torch',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=True, train_test_split=train_test_split)
    return test,train

class MetricsCalculator(pl.Callback):
    """自定义回调函数输出每个epoch的相关指标"""

    def __init__(self):
        super().__init__()
        self.best_miou = float('-inf')
        self.best_bmiou = float('-inf')
        self.best_combined = float('-inf')
        self.best_miou_epoch = -1
        self.best_bmiou_epoch = -1
        self.best_combined_epoch = -1

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """Lightning 会返回 Tensor 或 Python 数值，这里统一为 float"""
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return value.detach().float().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_metric(self, metrics, *names, default: float = 0.0) -> float:
        for name in names:
            if name in metrics:
                return self._to_float(metrics[name], default)
        return default

    def setup(self, trainer, pl_module, stage):
        """Lightning生命周期方法，在fit/test开始时调用"""
        # 这里初始化，保证在分布式环境中正确
        self.best_miou = float('-inf')
        self.best_bmiou = float('-inf')
        self.best_combined = float('-inf')
        self.best_miou_epoch = -1
        self.best_bmiou_epoch = -1
        self.best_combined_epoch = -1
        self.best_epoch = -1


    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics

        # 主模型训练指标
        train_acc = self._get_metric(metrics, 'train_acc', 'train_acc_epoch')
        train_miou = self._get_metric(metrics, 'train_miou', 'train_miou_epoch')
        train_bmiou = self._get_metric(metrics, 'train_bmiou', 'train_bmiou_epoch')
        train_boundary_loss = self._get_metric(metrics, 'train_boundary_loss', 'train_boundary_loss_epoch')
        train_seg_loss = self._get_metric(metrics, 'train_seg_loss', 'train_seg_loss_epoch')
        train_loss = self._get_metric(metrics, 'train_loss', 'train_loss_epoch')
        train_boundary_weight = self._get_metric(metrics, 'train_boundary_weight', 'train_boundary_weight_epoch')
        train_aux_total = self._get_metric(metrics, 'train_aux_total', 'train_aux_total_epoch')
        train_aux_local = self._get_metric(metrics, 'train_aux_local', 'train_aux_local_epoch')
        train_aux_mid = self._get_metric(metrics, 'train_aux_mid', 'train_aux_mid_epoch')
        train_aux_global = self._get_metric(metrics, 'train_aux_global', 'train_aux_global_epoch')
        train_aux_temp = self._get_metric(metrics, 'train_aux_temp', 'train_aux_temp_epoch')
        train_aux_fused = self._get_metric(metrics, 'train_aux_fused', 'train_aux_fused_epoch')
        # 输出主模型训练指标
        print(f"\n=== Epoch {trainer.current_epoch + 1} Training Metrics ===")
        print(f"[Main Model]")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Train mIoU: {train_miou:.4f}")
        print(f"  Train Boundary mIoU: {train_bmiou:.4f}")
        print(f"  Train Seg Loss: {train_seg_loss:.4f}")
        print(f"  Train Boundary Loss: {train_boundary_loss:.4f}")
        print(f"  Boundary Weight (active): {train_boundary_weight:.4f}")
        print(f"  Train Total Loss: {train_loss:.4f}")
        print(f"[Auxiliary Heads]")
        print(f"  Total Aux Loss: {train_aux_total:.4f}")
        print(f"    Local: {train_aux_local:.4f} | Mid: {train_aux_mid:.4f} | Global: {train_aux_global:.4f}")
        print(f"    Temp: {train_aux_temp:.4f} | Fused: {train_aux_fused:.4f}")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时的回调"""
        if not trainer.is_global_zero:
            return
    
        # 跳过 sanity check 阶段的输出
        if trainer.sanity_checking:
            return
        
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch + 1

        # 获取验证指标
        val_acc = self._get_metric(metrics, 'val_acc', 'val_acc_epoch')
        val_miou = self._get_metric(metrics, 'val_miou', 'val_miou_epoch')
        val_bmiou = self._get_metric(metrics, 'val_bmiou', 'val_bmiou_epoch')
        val_loss = self._get_metric(metrics, 'val_loss', 'val_loss_epoch')
        val_aux_total = self._get_metric(metrics, 'val_aux_total', 'val_aux_total_epoch')
        val_aux_local = self._get_metric(metrics, 'val_aux_local', 'val_aux_local_epoch')
        val_aux_mid = self._get_metric(metrics, 'val_aux_mid', 'val_aux_mid_epoch')
        val_aux_global = self._get_metric(metrics, 'val_aux_global', 'val_aux_global_epoch')
        val_aux_temp = self._get_metric(metrics, 'val_aux_temp', 'val_aux_temp_epoch')
        val_aux_fused = self._get_metric(metrics, 'val_aux_fused', 'val_aux_fused_epoch')

        # 计算组合指标（可以调整权重）
        val_combined = val_miou + val_bmiou

        # 更新最佳记录（仅用于显示）
        if val_miou > self.best_miou:
            self.best_miou = val_miou
            self.best_miou_epoch = current_epoch
        
        if val_bmiou > self.best_bmiou:
            self.best_bmiou = val_bmiou
            self.best_bmiou_epoch = current_epoch
        
        if val_combined > self.best_combined:
            self.best_combined = val_combined
            self.best_combined_epoch = current_epoch
        # 输出验证指标
        print(f"\n=== Epoch {current_epoch} Validation Metrics ===")
        print(f"[Main Model]")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val mIoU: {val_miou:.4f} (best: {self.best_miou:.4f} @ epoch {self.best_miou_epoch})")
        print(f"  Val Boundary mIoU: {val_bmiou:.4f} (best: {self.best_bmiou:.4f} @ epoch {self.best_bmiou_epoch})")
        print(f"  Val Combined: {val_combined:.4f} (best: {self.best_combined:.4f} @ epoch {self.best_combined_epoch})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"[Auxiliary Heads]")
        print(f"  Total Aux Loss: {val_aux_total:.4f}")
        print(f"    Local: {val_aux_local:.4f} | Mid: {val_aux_mid:.4f} | Global: {val_aux_global:.4f}")
        print(f"    Temp: {val_aux_temp:.4f} | Fused: {val_aux_fused:.4f}")
        print("=" * 50)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run testing')
    parser.add_argument('--epochs', type=int,
                        help='How many epochs to train', default=200)
    parser.add_argument('--tb_save_dir', type=str,
                        help='Tensorboard save directory', default='train_logs')
    parser.add_argument('--experiment_name', type=str,default="BAMSF",
                        help='Experiment Name')
    parser.add_argument('--experiment_version', type=str,
                        help='Experiment Version')
    parser.add_argument('--train_batch_size', type=int,
                        help='Train batch size', default=8 )
    parser.add_argument('--devices', nargs='+', help='Devices to use', default=[0,1])
    
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=2)
    parser.add_argument('--ckpt', type=str,help='Checkpoint path')
    parser.add_argument('--boundary_contrast_weight', type=float,
                        help='Max weight for boundary contrastive loss', default=1.2)
    parser.add_argument('--boundary_warmup_epochs', type=int,
                        help='Warmup epochs to ramp boundary loss weight', default=10)
    parser.add_argument('--boundary_contrast_nsample', type=int,
                        help='Neighbor count used in boundary contrastive loss', default=12)
    parser.add_argument('--boundary_contrast_temperature', type=float,
                        help='Temperature for boundary contrastive loss', default=0.07)
    parser.add_argument('--bmiou_k', type=int,
                        help='Neighbor size for boundary mIoU computation', default=12)
    parser.add_argument('--aux_local_weight', type=float, default=0.2,
                        help='Weight for local auxiliary supervision')
    parser.add_argument('--aux_mid_weight', type=float, default=0.2,
                        help='Weight for mid-level auxiliary supervision')
    parser.add_argument('--aux_global_weight', type=float, default=0.25,
                        help='Weight for global auxiliary supervision')
    parser.add_argument('--aux_temp_weight', type=float, default=0.15,
                        help='Weight for temp classifier supervision')
    parser.add_argument('--aux_fused_weight', type=float, default=0.3,
                        help='Weight for fused feature auxiliary supervision')
    parser.add_argument('--class_weights', type=str,
                        help='Comma separated class weights for loss balancing')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Enable focal loss for primary supervision')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                        help='Gamma value for focal loss')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate for cosine restarts')
    parser.add_argument('--lr_restart_interval', type=int, default=50,
                        help='Epoch interval for cosine annealing warm restarts')
    parser.add_argument('--lr_restart_mult', type=int, default=2,
                        help='Multiplier for cosine restart interval')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    class_weights = None
    if args.class_weights:
        try:
            class_weights = [float(w) for w in args.class_weights.split(',') if w.strip()]
        except ValueError as exc:
            raise ValueError('Failed to parse --class_weights, please provide comma separated floats') from exc

    test_dataset,train_dataset = get_dataset(args.train_test_split)

    model = LitDilatedToothSegmentationNetwork(
        boundary_contrast_weight=args.boundary_contrast_weight,
        boundary_warmup_epochs=args.boundary_warmup_epochs,
        aux_local_weight=args.aux_local_weight,
        aux_mid_weight=args.aux_mid_weight,
        aux_global_weight=args.aux_global_weight,
        aux_temp_weight=args.aux_temp_weight,
        aux_fused_weight=args.aux_fused_weight,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        boundary_contrast_nsample=args.boundary_contrast_nsample,
        boundary_contrast_temperature=args.boundary_contrast_temperature,
        bmiou_k=args.bmiou_k,
        lr_min=args.lr_min,
        lr_restart_interval=args.lr_restart_interval,
        lr_restart_mult=args.lr_restart_mult,
        weight_decay=args.weight_decay,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,  # 验证时减少workers
        pin_memory=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True  # 保持worker进程
    )
    if args.experiment_name is None:
        experiment_name = f'{args.model}_threedteethseg'
    else:
        experiment_name = args.experiment_name

    experiment_version = args.experiment_version

    logger = TensorBoardLogger(
        save_dir=args.tb_save_dir,
        name=experiment_name,
        version=experiment_version
    )

    log_dir = logger.log_dir
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    metrics_callback = MetricsCalculator()
    checkpoint_callback_miou = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_miou-{epoch:02d}-{val_miou:.4f}',
        monitor='val_miou',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    checkpoint_callback_bmiou = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_bmiou-{epoch:02d}-{val_bmiou:.4f}',
        monitor='val_bmiou',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    from lightning.pytorch.strategies import DDPStrategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='cuda',
        devices=[int(d) for d in args.devices],
        enable_progress_bar=True,
        logger=logger,
        precision=args.n_bit_precision,
        deterministic=False,
        callbacks=[metrics_callback],
        gradient_clip_val=1.0,            # ← 防爆梯度，验证更稳
        strategy=DDPStrategy(
            find_unused_parameters=False,
            timeout=timedelta(minutes=30),
            static_graph=False
        )
    )

    
    trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader, ckpt_path=args.ckpt)
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"Results saved to: {log_dir}")
    print(f"{'='*60}")
