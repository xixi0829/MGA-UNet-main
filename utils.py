import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import torch

# 尝试导入 thop，用于计算 FLOPs (可能会失败，但参数计算不受影响)
try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped.")
    THOP_AVAILABLE = False


def calculate_model_complexity(model, input_size=(3, 256, 256)):
    """
    计算模型的参数量 (Params) 和计算量 (FLOPs)。

    Args:
        model (nn.Module): 要计算的模型实例。
        input_size (tuple): 模型的输入尺寸，格式为 (C, H, W)。

    Returns:
        tuple: (params_M, flops_G) - 参数量（百万）, FLOPs（十亿）
    """
    # 1. 计算参数量 (Params)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_M = total_params / 1e6

    flops_G = 0.0

    # 2. 计算 FLOPs (需要 thop 库)
    if THOP_AVAILABLE:
        try:
            # 创建一个假输入张量
            dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)

            # 使用 profile 计算 FLOPs
            # 注意：Mamba 等复杂模块的 FLOPs 计算容易失败，需确保 thop 兼容性
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_G = flops / 1e9

        except Exception as e:
            # 如果计算失败，打印警告并跳过 FLOPs
            print(f"Warning: FLOPs calculation failed (Error: {e}). Skipping FLOPs.")
            flops_G = -1.0  # 用 -1 标记计算失败

    return params_M, flops_G


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        filename = f"{test_data_name}_{i}.png"
    else:
        filename = f"{i}.png"

    # 使用 os.path.join 确保路径正确
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss Implementation.
    alpha (alpha) 控制 FP 的惩罚 (误判前景)
    beta (beta) 控制 FN 的惩罚 (漏判前景 - 对应低 Sensitivity)
    建议设置 beta > alpha 来提高 Sensitivity。
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        size = pred.size(0)

        # 展平输入
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # 计算 TP, FP, FN
        TP = (pred_ * target_).sum(1)
        FP = ((1 - target_) * pred_).sum(1)  # 预测为 1，实际为 0
        FN = (target_ * (1 - pred_)).sum(1)  # 预测为 0，实际为 1 (漏判)

        # 计算 Tversky Score
        tversky_score = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Tversky Loss
        tversky_loss = 1 - tversky_score.mean()  # 对所有批次求平均损失

        return tversky_loss


# --- 修改：将 BceDiceLoss 重命名为 BceTverskyLoss ---
class BceTverskyLoss(nn.Module):  #
    # 修改：原有的 BceDiceLoss 现在使用 BCELoss 和 TverskyLoss
    def __init__(self, wb=1, wt=1, alpha=0.3, beta=0.7):  #
        super(BceTverskyLoss, self).__init__()
        self.bce = BCELoss()
        # 实例化 Tversky Loss，使用 alpha < 0.5 和 beta > 0.5 来提高 Sensitivity
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.wb = wb  # BCE 权重
        self.wt = wt  # Tversky 权重

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        tverskyloss = self.tversky(pred, target)

        # 非对称加权损失
        loss = self.wt * tverskyloss + self.wb * bceloss
        return loss
# --- 修改结束 ---