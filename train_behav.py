import os, random
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ramp import generate_ramp_target
from dataset import TimingDataset
from model import FiringRateRNN
from torch.utils.data import Dataset, DataLoader, Subset
from dataset import build_index_with_kfold, get_fold_dataloaders, collate_variable_length
from collections import defaultdict
from scipy.stats import pearsonr
import datetime
import time
from scipy.spatial import procrustes



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Save checkpoint
def save_checkpoint(model, save_path):
    """save the model checkpoint"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Recorder class
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)
        # plt.ion()  # interactive mode on
        # self.fig, self.ax = plt.subplots(figsize=(10, 5))

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, val_loss):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.current_epoch = idx + 1

    # def plot_dynamic(self):
    #     self.ax.clear()
    #     self.ax.grid()
    #     self.ax.set_title('Loss curve of train/val')
    #     self.ax.set_xlabel('Training epoch')
    #     self.ax.set_ylabel('Loss')

    #     x_axis = np.arange(self.current_epoch)  # x-axis for the current epoch
    #     self.ax.plot(x_axis, self.epoch_losses[:self.current_epoch, 0], color='g', linestyle='-', label='Train loss')
    #     self.ax.plot(x_axis, self.epoch_losses[:self.current_epoch, 1], color='y', linestyle='-', label='Validation loss')
    #     self.ax.legend(loc='best')
    #     plt.pause(0.01)  # pause for a short time to update the plot

    def plot_curve(self, save_path):
        # Plot the complete loss curve
        fig = plt.figure(figsize=(20, 10))
        x_axis = np.arange(self.total_epoch)
        plt.grid()
        plt.title('Loss curve of train/val')
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')

        plt.plot(x_axis, self.epoch_losses[:, 0], color='g', linestyle='-', label='Train loss')
        plt.plot(x_axis, self.epoch_losses[:, 1], color='y', linestyle='-', label='Validation loss')
        plt.legend(loc='best')
        fig.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        if self.log_txt_path:
            with open(self.log_txt_path, 'a') as f:
                f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def train(model, optimizer, train_loader, device, epoch, log_txt_path):
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    total_loss = 0.0

    for batch_idx, (u_batch, target_batch, set_idx_batch, dotN_batch, lengths, cond) in enumerate(train_loader):
        # u_batch: (batch, T_max, 2)
        # target_batch: (batch, T_max)
        # set_idx_batch: (batch,)
        # lengths: (batch,)

        u_batch = u_batch.to(device)
        target_batch = target_batch.to(device)
        set_idx_batch = set_idx_batch.to(device)
        lengths = lengths.to(device)

        outputs = model(u_batch, x0=None, return_all=True)  # (batch, T_max)
        loss = 0.0
        batch_size = u_batch.size(0)

        for i in range(batch_size): # for each sample in the batch
            set_idx_i = set_idx_batch[i].item()
            T_i = lengths[i].item() # original trial length
            pred = outputs[i, set_idx_i + 1:T_i]  # model output: production phase only
            target = target_batch[i, set_idx_i + 1:T_i] # ramp target: production phase only 
            loss += F.mse_loss(pred, target)
        loss = loss / batch_size # average loss over the batch

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: NaN or Inf detected in gradients at {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.update(loss.item(), batch_size)
        progress.display(batch_idx)

        total_loss += loss.item()
    return total_loss / len(train_loader)  # average loss over all batches


def validate(model, val_loader, device, epoch, log_txt_path):
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(val_loader),
                             [losses],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (u_batch, target_batch, set_idx_batch, dotN_batch, lengths,cond) in enumerate(val_loader):

            u_batch = u_batch.to(device)
            target_batch = target_batch.to(device)
            set_idx_batch = set_idx_batch.to(device)
            lengths = lengths.to(device)

            outputs = model(u_batch, x0=None, return_all=True)
            loss = 0.0
            batch_size = u_batch.size(0)
            for i in range(batch_size):
                set_idx_i = set_idx_batch[i].item()
                T_i = lengths[i].item()
                pred = outputs[i, set_idx_i + 1:T_i]
                target = target_batch[i, set_idx_i + 1:T_i]
                loss += F.mse_loss(pred, target)
            loss = loss / batch_size

            losses.update(loss.item(), batch_size)
            progress.display(batch_idx)

            total_loss += loss.item()

    return total_loss / len(val_loader)  # average loss over all batches



def compute_condition_means(model, loader, device, threshold=1.0):
    model.eval()
    preds = defaultdict(list)
    trues = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x       = batch[0].to(device)           # 输入
            true_tp = true_tp = batch[4].cpu().numpy() - batch[2].cpu().numpy() # 真实 Tp
            cond    = batch[5].cpu().numpy()        # condition id

            # raw_out 假设 shape = (B, T)
            raw_out = model(x).cpu().numpy()

            # 把每条序列映射到一个标量：第一个超过阈值的位置
            # 如果从未超过，就返回最后一个时间点
            y_pred = []
            for seq in raw_out:
                # 找到 seq >= threshold 的所有索引
                exceed = np.where(seq >= threshold)[0]
                if exceed.size > 0:
                    y_pred.append(int(exceed[0]))
                else:
                    y_pred.append(seq.shape[0] - 1)
            y_pred = np.array(y_pred)  # 变成 (B,)

            # 现在 y_pred, true_tp, cond 都是一维向量了
            for p, t, c in zip(y_pred, true_tp, cond):
                preds[int(c)].append(p)
                trues[int(c)].append(t)

    # 计算每个 condition 下的均值
    pred_means = []
    true_means = []
    for i in range(max(preds.keys()) + 1):
        if preds[i] and trues[i]:
            pred_means.append(np.mean(preds[i]))
            true_means.append(np.mean(trues[i]))
    pred_means = np.array(pred_means)
    true_means = np.array(true_means)
    return pred_means, true_means

# def condition_metric_pearson(pred_means, true_means):
#     if len(pred_means) < 2:
#         return np.inf
#     r, _ = pearsonr(pred_means, true_means)
#     return 1 - r

def condition_metric_procrustes(pred_means, true_means):
    if len(pred_means) < 2:
        return np.inf
    # reshape 成二维 (n, 1) 矩阵，因为 Procrustes 要求是 shape=(n, dim)
    mtx1 = np.array(true_means).reshape(-1, 1)
    mtx2 = np.array(pred_means).reshape(-1, 1)

    _, _, disparity = procrustes(mtx1, mtx2)
    return disparity  # 越小表示越相似


# ================== #

# Model training
def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Environment setup
    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    project_path = './'
    log_dir = os.path.join(project_path, 'log')
    model_dir = os.path.join(project_path, 'model')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_txt_path = os.path.join(log_dir, f"{time_str}fold{config['fold']}-log.txt")
    curve_save_path = os.path.join(log_dir, f"{time_str}fold{config['fold']}-curve.png")
    model_save_path = os.path.join(model_dir, f"{time_str}fold{config['fold']}-best_model.pt")
    fold_indices_path = os.path.join(project_path, "fold_indices.json")

    device = config['device'] if torch.cuda.is_available() else 'cpu'

    with open(log_txt_path, 'a') as f:
        f.write("log information\n\n")

    # Load model and data
    model = FiringRateRNN(hidden_size=200, input_dim=5).to(device)
    trials_list = torch.load(config['data_path'])
    dataset = TimingDataset(trials_list, device=device)

    #index_sets = build_index_with_kfold(trials_list, n_splits=5, random_state=SEED, save_path=fold_indices_path)
    index_sets = build_index_with_kfold(trials_list, n_splits=5, random_state=SEED,save_path=None)
    train_loader, val_loader = get_fold_dataloaders(dataset, index_sets, fold=config['fold'], batch_size=config['batch_size'], collate_fn=collate_variable_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training
    recorder = RecorderMeter(total_epoch=config['epochs'])
    best_val_loss = float('inf')
    best_metric = float('inf')
    metric_list = []


    with open(log_txt_path, 'a') as f:
        f.write(f"dataset: {config['data_path']}\n")
        f.write(f"Training with fold {config['fold']}\n")
        f.write(f"random seed: {SEED}\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Initial Learning rate: {config['learning_rate']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Device: {device}\n")
        f.write(f"step size: {scheduler.step_size}\n")
        f.write(f"gamma: {scheduler.gamma}\n\n")

    
    
    
    
    
    # 训练循环开始前，先初始化这两个“Best”指标
    best_val_loss = float("inf")
    best_metric   = float("inf")

    for epoch in range(config['epochs']):

        inf = f'******************** {epoch} ********************'
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        train_loss = train(model, optimizer, train_loader, device, epoch, log_txt_path)
        val_loss = validate(model, val_loader, device, epoch, log_txt_path)

        # —— 只算一次 metric，并记录 —— 
        pred_means, true_means = compute_condition_means(model, val_loader, device)
        metric = condition_metric_procrustes(pred_means, true_means)
        metric_list.append(metric)
        with open(log_txt_path, 'a') as f:
            f.write(f"Epoch {epoch}: disparity = {metric:.4f}\n")

        recorder.update(epoch, train_loss, val_loss)

        best_model_str = ""

        # if val_loss <= 0.03:
        # # if val_loss < 0.01:
        # # 只用 metric（disparity）来判断
        #     if metric < best_metric:
        #         best_metric = metric
        #         save_checkpoint(model, model_save_path)
        #         with open(log_txt_path, 'a') as f:
        #             f.write(f"Saved by METRIC at epoch {epoch}, disparity={metric:.4f}\n")
        #     else:
        #         with open(log_txt_path, 'a') as f:
        #             f.write(f"skip saving: disparity={metric:.4f} ≥ best_metric={best_metric:.4f}\n")
        # elif val_loss < best_val_loss:
        #     # loss ≥ 0.03 时，才考虑 loss
        #     best_val_loss = val_loss
        #     save_checkpoint(model, model_save_path)
        #     with open(log_txt_path, 'a') as f:
        #         f.write(f"Saved by LOSS at epoch {epoch}, val_loss={val_loss:.4f}\n")
        # else:
        #     with open(log_txt_path, 'a') as f:
        #         f.write(f"skip saving: val_loss={val_loss:.4f} ≥ best_loss={best_val_loss:.4f}\n")
        
        # scheduler.step()

        # -------------------- 6.3改 -------------------- #
        if val_loss > 0.03:
                # --- loss 驱动分支 ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metric   = metric   # 同时记录当前模型对应的 metric
                    save_checkpoint(model, model_save_path)
                    with open(log_txt_path, 'a') as f:
                        f.write(f"Saved by LOSS at epoch {epoch}, val_loss={val_loss:.4f}, disparity={metric:.4f}\n")
                else:
                    with open(log_txt_path, 'a') as f:
                        f.write(f"skip saving: val_loss={val_loss:.4f} ≥ best_val_loss={best_val_loss:.4f}\n")

        else:
                # --- loss ≤ 0.03，换成 metric（disparity）驱动 ---
                if metric < best_metric:
                    best_val_loss = val_loss
                    best_metric   = metric
                    save_checkpoint(model, model_save_path)
                    with open(log_txt_path, 'a') as f:
                        f.write(f"Saved by METRIC at epoch {epoch}, val_loss={val_loss:.4f}, disparity={metric:.4f}\n")
                else:
                    with open(log_txt_path, 'a') as f:
                        f.write(f"skip saving: disparity={metric:.4f} ≥ best_metric={best_metric:.4f}\n")

        # 更新学习率调度器
        scheduler.step()

        # Get the updated learning rate
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        lr_info = f"Current learning rate: {current_learning_rate}\n"
        loss_info = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}\n"

        # Print and log the updated learning rate
        #print(inf)
        print(lr_info.strip())
        print(loss_info.strip())
        if best_model_str:
            print(best_model_str.strip())

        with open(log_txt_path, 'a') as f:
            # f.write(inf + '\n')
            f.write(lr_info)
            f.write(loss_info)
            if best_model_str:
                f.write(best_model_str)

    recorder.plot_curve(save_path=curve_save_path)
    print(f"Loss curve saved to {curve_save_path}")


if __name__ == '__main__':
    main()





        
