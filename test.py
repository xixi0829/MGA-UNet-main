import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *


from models.MGA_UNet import MGA_UNet

from engine import *
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = config.test_weights  # 从配置中获取路径

    # 确保 config.test_weights 包含完整的路径名
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    # gpu_ids = [0]# [0, 1, 2, 3] # --- 修改：此行不再需要用于 DataParallel ---
    torch.cuda.empty_cache()

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = MGA_UNet(num_classes=model_cfg['num_classes'],
                          input_channels=model_cfg['input_channels'],
                          c_list=model_cfg['c_list'],
                          # split_att=model_cfg['split_att'],
                          bridge=model_cfg['bridge']
                          )

    # --- 修改：移除 DataParallel，直接将模型移至 GPU ---
    model = model.cuda()
    # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0]) #
    # --- 修改结束 ---

    print('#----------Preparing dataset----------#')
    # 使用 test_mode=True 来加载整个测试集
    test_dataset = isic_loader(path_Data=config.data_path, test_mode=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,  # (使用 config 中的 num_workers)
                             drop_last=False)  # 测试时不要丢弃任何样本

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))

    # 检查模型权重是否被 DataParallel 封装 (如果存在 'module.' 前缀)
    if 'state_dict' in best_weight:
        best_weight = best_weight['state_dict']

    # --- 新增过滤函数 (保持不变，它对于加载旧权重很有用) ---
    def clean_state_dict(state_dict):
        """移除所有与 FLOPs/参数统计相关的意外键"""
        new_state_dict = {}
        for k, v in state_dict.items():
            # 过滤掉包含 'total_ops' 或 'total_params' 的键
            if 'total_ops' not in k and 'total_params' not in k:
                # 移除 DataParallel 引入的 'module.' 前缀 (如果存在)
                if k.startswith('module.'):  #
                    k = k[7:]  #

                # 过滤掉 ln_out.weight/bias (仅在旧版本或不同分支中存在)
                if 'ln_out.' not in k:
                    new_state_dict[k] = v
        return new_state_dict

    # --- 使用过滤后的权重加载 ---
    best_weight = clean_state_dict(best_weight)
    # 注意：这里直接使用了 clean_state_dict，这会解决 total_ops 和 DataParallel 的问题

    # --- 修改：移除 .module ---
    model.load_state_dict(best_weight, strict=False)  # 使用 strict=False 忽略结构差异较大的键
    # --- 修改结束 ---

    loss = test_one_epoch(
        test_loader,
        model,
        criterion,
        logger,
        config,
    )


if __name__ == '__main__':
    config = setting_config
    main(config)