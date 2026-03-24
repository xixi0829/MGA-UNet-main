from torchvision import transforms
from utils import *

from datetime import datetime


class setting_config:
    """
    the config of training setting.
    """
    network = 'MGA_UNet'
    # network = 'MGA_Net1'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8, 16, 24, 32, 48, 64],  # <-- 6 层模型
        'bridge': True,
    }

    test_weights = r'results/isic2017_best.pth'

    datasets = 'ISIC20181'
    if datasets == 'ISIC20181':
        data_path = r'E:\data\ISIC20181'
    elif datasets == 'DSB2018':
        data_path = r'D:\data\DSB2018'
    elif datasets == 'PH2':
        data_path = r'D:\data\PH2'
    elif datasets == 'Polyp':
        data_path = r'D:\data\Polyp'
    elif datasets == 'ISIC2017':
        data_path = r'D:\data\ISIC2017/train'
    else:
        raise Exception('datasets in not right!')

    # **************************** Tversky Loss 参数 ****************************
    # Tversky Loss (wt) 权重
    weight_tversky = 3.0
    # BCE Loss (wb) 权重
    weight_bce = 0.5
    k_folds = 5
    # Tversky alpha: 惩罚 FP (误判前景，对应低 Specificity)
    tversky_alpha = 0.4
    # Tversky beta: 惩罚 FN (漏判前景，对应低 Sensitivity)
    tversky_beta = 0.6

    criterion = BceTverskyLoss(wb=weight_bce,
                               wt=weight_tversky,
                               alpha=tversky_alpha,
                               beta=tversky_beta)
    # *************************************************************************

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1

    num_workers = 0  # <-- 关键参数 (慢，但mIoU最高)
    seed = 42

    world_size = None
    rank = None
    amp = False
    batch_size = 16
    epochs = 200
    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

    print_interval = 60
    val_interval = 1
    save_interval = 10
    threshold = 0.5

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 0.002
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 0.002  # <-- 关键参数
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.05  # <-- 注意: 这个日志 的 weight_decay 是 0.03
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False

    sch = 'WP_CosineLR'  # <-- 关键参数
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = 80
        eta_min = 0.00001
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 25
        T_mult = 1
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 30  # <-- 关键参数