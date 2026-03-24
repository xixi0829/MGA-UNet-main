import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import numpy as np
import warnings

# 引入你的模块
from loader import isic_loader
from models.MGA_UNet import MGA_UNet
from engine import train_one_epoch, val_one_epoch
from utils import get_logger, log_config_info, set_seed, calculate_model_complexity, get_optimizer, get_scheduler
from configs.config_setting import setting_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def main(config):
    # 全局随机种子设置 (保证每次运行的划分是一样的)
    set_seed(config.seed)

    # 记录每一折的最佳结果
    fold_best_mious = []

    print(f'#----------Starting {config.k_folds}-Fold Cross Validation----------#')

    # --- 五折交叉验证循环 ---
    for fold in range(config.k_folds):
        print(f'\n#========== Fold {fold} / {config.k_folds - 1} ==========#')

        # 1. 为每一折创建独立的目录
        config.work_dir = os.path.join(config.work_dir, f'Fold_{fold}')
        log_dir = os.path.join(config.work_dir, 'log')
        checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
        outputs = os.path.join(config.work_dir, 'outputs')

        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        if not os.path.exists(outputs): os.makedirs(outputs)

        # 2. 获取 Logger
        # 注意：这里不需要 sys.path.append，否则路径会越来越长
        logger = get_logger(f'train_fold_{fold}', log_dir)
        log_config_info(config, logger)

        # 3. 清理 GPU 缓存，确保显存干净
        torch.cuda.empty_cache()

        # 4. 准备数据 (关键：传入 fold_idx)
        print('#----------Preparing dataset----------#')
        train_dataset = isic_loader(path_Data=config.data_path, train=True, fold_idx=fold, num_folds=config.k_folds,
                                    seed=config.seed)
        val_dataset = isic_loader(path_Data=config.data_path, train=False, fold_idx=fold, num_folds=config.k_folds,
                                  seed=config.seed)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True,
                                num_workers=config.num_workers, drop_last=False)

        # 5. 重新初始化模型 (关键：必须重新实例化，否则会继承上一折的权重！)
        print('#----------Preparing Model (Reset)----------#')
        model_cfg = config.model_config
        model = MGA_UNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'])
        model = model.cuda()

        # 打印一次复杂度即可
        if fold == 0:
            input_size = (config.input_channels, config.input_size_h, config.input_size_w)
            params_M, flops_G = calculate_model_complexity(model, input_size)
            logger.info(f'Model Complexity: Params: {params_M:.4f}M, FLOPs: {flops_G:.4f}G')

        # 6. 重新初始化优化器和调度器
        criterion = config.criterion
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        scaler = GradScaler()

        max_miou = 0.0
        best_epoch = 1

        # 7. 开始训练 Loop
        for epoch in range(1, config.epochs + 1):
            torch.cuda.empty_cache()

            train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)

            # 验证
            loss, miou, f1, spe, sen, acc = val_one_epoch(val_loader, model, criterion, epoch, logger, config)

            # 保存最佳
            if miou > max_miou:
                max_miou = miou
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                logger.info(f'Fold {fold} Best Epoch: {epoch} with mIoU: {miou:.4f}')

            # 保存最新
            torch.save({
                'epoch': epoch,
                'max_miou': max_miou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        # 8. 记录本折结果
        print(f'Fold {fold} Finished. Best mIoU: {max_miou:.4f}')
        fold_best_mious.append(max_miou)

        # 重命名最佳权重
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-fold{fold}-epoch{best_epoch}-miou{max_miou:.4f}.pth')
            )

    # --- 所有折结束，打印汇总 ---
    print('\n#========== 5-Fold CV Finished ==========#')
    print(f'Scores per fold: {fold_best_mious}')
    mean_miou = np.mean(fold_best_mious)
    std_miou = np.std(fold_best_mious)
    print(f'Final Result: Mean mIoU = {mean_miou:.4f} ± {std_miou:.4f}')

    # 将最终结果写入根目录的日志
    with open(os.path.join(config.work_dir, 'final_result.txt'), 'w') as f:
        f.write(f'Scores per fold: {fold_best_mious}\n')
        f.write(f'Mean mIoU: {mean_miou:.4f}\n')
        f.write(f'Std mIoU: {std_miou:.4f}\n')


if __name__ == '__main__':
    config = setting_config
    # 创建基础目录
    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir)
    main(config)