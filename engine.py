import os
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from utils import save_imgs

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                out = model(images)
            with autocast(enabled=False):
                loss = criterion(out.float(), targets.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    scheduler.step()
    return np.mean(loss_list)

def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    loss_list = []

    # 修改：改用全局计数器，避免单张图极端值拉低平均分
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Val Epoch {epoch}"):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            if isinstance(out, tuple): out = out[0]

            out_np = (out.squeeze(1).cpu().detach().numpy() >= config.threshold).astype(int)
            msk_np = (msk.squeeze(1).cpu().detach().numpy() >= 0.5).astype(int)

            # 累加整个 Batch 的混淆矩阵元素
            total_tp += np.sum((out_np == 1) & (msk_np == 1))
            total_fp += np.sum((out_np == 1) & (msk_np == 0))
            total_fn += np.sum((out_np == 0) & (msk_np == 1))
            total_tn += np.sum((out_np == 0) & (msk_np == 0))

    smooth = 1e-6
    # 计算全局指标
    m_miou = total_tp / (total_tp + total_fp + total_fn + smooth)
    m_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + smooth)
    m_sen = total_tp / (total_tp + total_fn + smooth)
    m_spe = total_tn / (total_tn + total_fp + smooth)
    m_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + smooth)
    m_loss = np.mean(loss_list)

    log_info = f'val epoch: {epoch}, loss: {m_loss:.4f}, miou: {m_miou:.4f}, Dice: {m_dice:.4f}, Spe: {m_spe:.4f}, Sen: {m_sen:.4f}, Acc: {m_acc:.4f}'
    print(log_info)
    logger.info(log_info)

    return m_loss, m_miou, m_dice, m_spe, m_sen, m_acc

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    loss_list = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    output_dir = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            out_img = out.squeeze(1).cpu().detach().numpy()
            msk_img = msk.squeeze(1).cpu().detach().numpy()

            save_imgs(img, msk_img, out_img, i, output_dir, config.datasets, config.threshold, test_data_name=test_data_name)

            out_bin = (out_img >= config.threshold).astype(int)
            msk_bin = (msk_img >= 0.5).astype(int)

            total_tp += np.sum((out_bin == 1) & (msk_bin == 1))
            total_fp += np.sum((out_bin == 1) & (msk_bin == 0))
            total_fn += np.sum((out_bin == 0) & (msk_bin == 1))
            total_tn += np.sum((out_bin == 0) & (msk_bin == 0))

    smooth = 1e-6
    m_miou = total_tp / (total_tp + total_fp + total_fn + smooth)
    m_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + smooth)
    m_sen = total_tp / (total_tp + total_fn + smooth)
    m_spe = total_tn / (total_tn + total_fp + smooth)
    m_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + smooth)

    log_info = f'Test Result: Loss: {np.mean(loss_list):.4f}, mIoU: {m_miou:.4f}, Dice: {m_dice:.4f}, Acc: {m_acc:.4f}, Spe: {m_spe:.4f}, Sen: {m_sen:.4f}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)