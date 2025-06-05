
# %%
import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_path = '/Users/juntao/Desktop/proj_TimePerception/log/[05-07]-[08-29]-fold3-log.txt'

train_losses = []
val_losses = []
best_epochs = []

# 记录已经处理过的epoch，防止重复
seen_epochs = set()

with open(log_path, 'r') as f:
    for line in f:
        # 匹配训练和验证loss
        match = re.match(r"Epoch (\d+): train_loss=([0-9.]+), val_loss=([0-9.]+)", line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            # 只记录第一次出现的epoch
            if epoch not in seen_epochs:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                seen_epochs.add(epoch)
        
        # 检查是否是best model保存
        if "Best model saved at epoch" in line:
            best_epoch = int(re.search(r"epoch (\d+)", line).group(1))
            best_epochs.append(best_epoch)

# 打印提取结果确认
print(f"总共提取了 {len(train_losses)} 个epoch的数据")
print(f"Best model保存发生在这些epoch: {best_epochs}")

# 画图
epochs = np.arange(len(train_losses))

plt.figure(figsize=(12,6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s')

# 标出Best model保存的位置
for best_epoch in best_epochs:
    plt.scatter(best_epoch, val_losses[best_epoch], marker='*', s=200, c='red', label='Best Model' if best_epoch == best_epochs[0] else "")

plt.title('Training and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.xticks(epochs)
plt.tight_layout()

plt.show()

# 如果想保存，加这句
# plt.savefig('train_val_loss_curve_fixed.png', dpi=300)

# %%
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# 加载数据
dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_features.pt")


# 配色方案
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}

# 按条件(ts, dotN)分组，收集 tp
tp_by_condition = defaultdict(list)

for sample in dataset:
    ts = sample["ts"]
    dotN = sample["dotN"]
    seq_len = sample["seq_len"]
    set_idx = sample["set_idx"]
    label = sample["label"]
    tp = seq_len - set_idx

    key = (ts, dotN)
    tp_by_condition[key].append(tp)

# 计算每个条件下的平均 Tp 和用于绘图的 ts
ts_list = []
tp_list = []
labels = []

for (ts, dotN), tp_vals in sorted(tp_by_condition.items()):
    ts_list.append(ts)
    tp_list.append(sum(tp_vals) / len(tp_vals))
    labels.append(f"{ts/60:.2f}s, dotN={dotN}")

# 画图
plt.figure(figsize=(5, 4))
scatter = plt.scatter(ts_list, tp_list, c=range(len(ts_list)), cmap='plasma', s=80)

# 虚线：理想对角线
plt.plot([min(ts_list), max(ts_list)], [min(ts_list), max(ts_list)], 'k--')

plt.xlabel("Trained interval (frames)")
plt.ylabel("Tp (frames)")
plt.title("Tp vs Trained interval")

# 添加图例
handles, _ = scatter.legend_elements(prop="colors")
plt.legend(handles, labels, title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %%
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

# 加载数据
dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_features.pt", map_location="cpu")

# 2) 配色与条件映射
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}
# 顺序对应 label 0-7
duration_seconds = [2.48, 2.48, 3.98, 3.98, 4.48, 4.48, 4.98, 4.98]
dotN_list        = [   6,    11,    6,    11,    6,    11,    6,     11]

# 3) 构造排序后的 label 列表
labels_sorted = sorted(range(8), key=lambda l: (duration_seconds[l], dotN_list[l]))

# 4) 按 label 聚合 ts 和 tp
ts_by_label = defaultdict(list)
tp_by_label = defaultdict(list)
for sample in dataset:
    lbl      = sample["label"]
    ts_frame = sample["ts"]
    tp       = sample["seq_len"] - sample["set_idx"]
    ts_by_label[lbl].append(ts_frame)
    tp_by_label[lbl].append(tp)

# 5) 绘图
plt.figure(figsize=(5, 5))
for lbl in labels_sorted:
    ts_val   = ts_by_label[lbl][0]  # 同一 label ts 相同，取第一个
    avg_tp   = sum(tp_by_label[lbl]) / len(tp_by_label[lbl])
    dur_sec  = duration_seconds[lbl]
    dotN     = dotN_list[lbl]
    color    = duration_to_color[dur_sec]
    linestyle = '--' if dotN == 6 else '-'
    alpha     = 0.5  if dotN == 6 else 0.2

    # 绘制点
    plt.plot(ts_val, avg_tp,
             marker='o',
             color=color,
             linestyle=linestyle,
             alpha=alpha,
             markersize=8)

# 对角线（理想复现）
min_val = min(ts_by_label[l][0] for l in labels_sorted)
max_val = max(ts_by_label[l][0] for l in labels_sorted)
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

# 美化
plt.xlabel("Trained interval (frames)")
plt.ylabel("True Tp (frames)")
plt.title("Tp vs Trained interval")
plt.axis("square")
plt.grid(True)

# 自定义图例
legend_elems = [
    Line2D([0], [0], marker='o', color='purple', label='2.48s', linestyle=''),
    Line2D([0], [0], marker='o', color='navy',  label='3.98s', linestyle=''),
    Line2D([0], [0], marker='o', color='green', label='4.48s', linestyle=''),
    Line2D([0], [0], marker='o', color='gold',  label='4.98s', linestyle=''),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', alpha=0.5, label='dotN=6'),
    Line2D([0], [0], color='gray', lw=2, linestyle='-',  alpha=0.2, label='dotN=11'),
]
plt.legend(handles=legend_elems, title="Conditions",
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize='small', frameon=False)

plt.tight_layout()
plt.show()









# %%
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 读取你的保存数据
with open("outputs/inference_fold3_A_1_5.pkl", "rb") as f:
    results = pickle.load(f)

# 随机选两个神经元
np.random.seed(42)
units = np.random.choice(200, size=2, replace=False)
print("选择的神经元编号：", units)

# 提前设定颜色
colors = plt.cm.viridis(np.linspace(0, 1, 8))

for unit in units:
    # 每个label的数据分组
    label_firing = {label: [] for label in range(8)}

    for trial in results:
        firing = trial["firing_rate"]  # (T, 200)
        label = trial["label"]
        set_idx = trial["set_idx"]

        # 只取 set 后的 firing rate
        aligned_firing = firing[set_idx+1:, unit]  # (T_post,)
        label_firing[label].append(aligned_firing)

    plt.figure(figsize=(8, 5))

    for label in range(8):
        # 对不同长度的 trial，先对齐（补 NaN）
        max_len = max([len(arr) for arr in label_firing[label]])
        aligned = np.full((len(label_firing[label]), max_len), np.nan)
        for i, arr in enumerate(label_firing[label]):
            aligned[i, :len(arr)] = arr

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)

        x = np.arange(max_len)
        plt.plot(x, mean, color=colors[label], label=f'Label {label}')
        plt.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.2)

    plt.title(f'Unit {unit} firing rates across trials (Set-aligned)')
    plt.xlabel('Time since Set (frames)')
    plt.ylabel('Firing rate (tanh(x))')
    plt.legend()
    plt.show()

# for unit in units:
#     plt.figure(figsize=(8,4))
#     for trial in results:
#         fr = trial["firing_rate"][:, unit]   # (T,)
#         set_idx = trial["set_idx"]
#         label = trial["label"]
#         # 不同条件用不同颜色
#         plt.plot(fr, color=colors[label], alpha=0.3)
#     # 标出 Set 时刻
#     plt.axvline(set_idx, color='k', linestyle='--', lw=1)
#     plt.title(f"Unit {unit} full firing (pre- & post-Set)")
#     plt.xlabel("Time step")
#     plt.ylabel("Firing rate")
#     plt.show()

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
# 读取你的保存数据
with open("outputs/inference_fold3_dotN_A_1_5_beta_1_5.pkl", "rb") as f:
    results = pickle.load(f)

# 1) 按 label 收集 Tp 和 ts
label_Tp = {l: [] for l in range(8)}
label_ts = {l: [] for l in range(8)}

for entry in results:
    Tp    = entry["Tp"]
    ts    = entry["ts"]
    label = entry["label"]
    # 跳过没有 Tp 的 trial
    if Tp is None:
        continue
    label_Tp[label].append(Tp)
    label_ts[label].append(ts)

# 2) 计算每个 label 的平均 Tp 和 ts
mean_Tp = []
mean_ts = []
for l in range(8):
    if len(label_Tp[l]) == 0:
        # 没数据的话用 nan 占位
        mean_Tp.append(np.nan)
        mean_ts.append(np.nan)
    else:
        mean_Tp.append(np.mean(label_Tp[l]))
        mean_ts.append(np.mean(label_ts[l]))

# 3) 画图
plt.figure(figsize=(4,4))
colors = plt.cm.plasma(np.linspace(0,1,8))

# 散点
for l in range(8):
    plt.scatter(mean_ts[l], mean_Tp[l], color=colors[l], s=80, label=f"Cond {l}")

# 45° 参考线
# 为了让参考线覆盖所有有效点，这里只用非 nan 值计算范围
valid_x = [x for x in mean_ts if not np.isnan(x)]
valid_y = [y for y in mean_Tp if not np.isnan(y)]
mn = min(valid_x + valid_y)
mx = max(valid_x + valid_y)
plt.plot([mn, mx], [mn, mx], 'k--')

plt.xlabel("Trained interval (frames)")
plt.ylabel("Tp (frames)")
plt.title("Tp vs Trained interval")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()







# 新的颜色方案
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取你的保存数据
# with open("outputs/inference_fold3_dotN_A_1_5_beta_1_0_max.pkl", "rb") as f:
#     results = pickle.load(f)

with open("outputs/inference_dotN_features.pkl", "rb") as f:
    data = pickle.load(f)

results = data['inference']

# 随机选两个神经元
np.random.seed(42)
units = np.random.choice(200, size=2, replace=False)
print("Selected units:", units)

# 定义四种底色：对应四个 duration
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}

# 假设你的 label = 0..7 分别对应：
# durations = [2.48,2.48,3.98,3.98,4.48,4.48,4.98,4.98]
# dotNs      = [   6,   11,   6,   11,   6,   11,   6,    11]
duration_list = [2.48, 2.48, 3.98, 3.98, 4.48, 4.48, 4.98, 4.98]
dotN_list     = [   6,    11,    6,    11,    6,    11,    6,     11]

def lighten_color(color, amount=0.5):
    """Return a lighter shade of the given color."""
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple(rgb + (white - rgb) * amount)

for unit in units:
    # 每个 label 的 firing rate 列表
    label_firing = {label: [] for label in range(8)}

    for trial in results:
        firing = trial["firing_rate"]  # (T, 200)
        label = trial["label"]
        set_idx = trial["set_idx"]
        aligned = firing[set_idx+1:, unit]
        label_firing[label].append(aligned)

    plt.figure(figsize=(8, 5))
    for label in range(8):
        arrs = label_firing[label]
        if len(arrs) == 0:
            continue
        # 对齐不同长度的 trial（补 NaN）
        max_len = max(len(a) for a in arrs)
        aligned = np.full((len(arrs), max_len), np.nan)
        for i, a in enumerate(arrs):
            aligned[i, :len(a)] = a

        mean = np.nanmean(aligned, axis=0)
        std  = np.nanstd(aligned, axis=0)
        x = np.arange(len(mean))

        # 根据 duration 和 dotN 选择颜色、线型和透明度
        dur = duration_list[label]
        dotN = dotN_list[label]
        base_color = duration_to_color[dur]

        if dotN == 6:
            color     = base_color
            linestyle = '--'
            alpha     = 0.25
        else:
            color     = lighten_color(base_color, amount=0.6)
            linestyle = '-'
            alpha     = 0.2

        plt.plot(x, mean,
                 color=color,
                 linestyle=linestyle,
                 label=f'{dur}s, dotN={dotN}')
        plt.fill_between(x,
                         mean - std,
                         mean + std,
                         color=color,
                         alpha=alpha)

    plt.title(f'Unit {unit} firing rates across trials (Set-aligned)')
    plt.xlabel('Time since Set (frames)')
    plt.ylabel('Firing rate (tanh(x))')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Dur以及DOT N 6 和 11 的 Tp 对比
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 读取你的保存数据
with open("outputs/inference_stim_5c_behav_fold3.pkl", "rb") as f:
    data = pickle.load(f)

results = data['inference']

# with open("outputs/inference_with_setmask_fold3_A_3.pkl", "rb") as f:
#     results = pickle.load(f)

# 1) 按 label 收集 Tp 和 ts
label_Tp = {l: [] for l in range(8)}
label_ts = {l: [] for l in range(8)}
for entry in results:
    Tp    = entry["Tp"]
    ts    = entry["ts"]
    label = entry["label"]
    if Tp is None:
        continue
    label_Tp[label].append(Tp)
    label_ts[label].append(ts)

# 2) 计算每个 label 的平均 Tp 和 ts
mean_Tp = []
mean_ts = []
for l in range(8):
    if len(label_Tp[l]) == 0:
        mean_Tp.append(np.nan)
        mean_ts.append(np.nan)
    else:
        mean_Tp.append(np.mean(label_Tp[l]))
        mean_ts.append(np.mean(label_ts[l]))

# —— 以下是改动的部分 —— #

# 3a) 定义 duration→颜色 映射
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',   # 墨蓝
    4.48: 'green',
    4.98: 'gold'
}

# 3b) 定义 label→(duration, dotN)
duration_list = [2.48, 2.48, 3.98, 3.98, 4.48, 4.48, 4.98, 4.98]
dotN_list     = [   6,    11,    6,    11,    6,    11,    6,     11]

# 3c) 画图
fig = plt.figure(figsize=(5,5))  # 画布本身也保持正方形
ax = fig.add_subplot(1,1,1)

for l in range(8):
    x = mean_ts[l]
    y = mean_Tp[l]
    if np.isnan(x) or np.isnan(y):
        continue

    dur  = duration_list[l]
    dotN = dotN_list[l]
    color = duration_to_color[dur]
    alpha = 0.6 if dotN == 6 else 0.3

    ax.scatter(x, y,
               color=color,
               alpha=alpha,
               s=80,
               edgecolors='none',      # 去掉边缘描边
               label=f'{dur}s, dotN={dotN}')

# 45° 参考线
valid_x = [x for x in mean_ts if not np.isnan(x)]
valid_y = [y for y in mean_Tp if not np.isnan(y)]
mn = min(valid_x + valid_y)
mx = max(valid_x + valid_y)
ax.plot([mn, mx], [mn, mx], 'k--')

# 设置坐标轴等宽（正方形）
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("Trained interval (frames)")
ax.set_ylabel("Tp (frames)")
ax.set_title("Tp vs Trained interval")

# 缩小图例字体和标记
ax.legend(bbox_to_anchor=(1.05,1),
          loc='upper left',
          fontsize='small',
          markerscale=0.8,
          frameon=False)

plt.tight_layout()
plt.show()

# %%
# Z 序列随时间变化的图
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1) 加载数据
with open("outputs/inference_stim_5c_behav_fold3.pkl", "rb") as f:
    data = pickle.load(f)

label_z = data['z_timecourses']  # dict: label -> {'time', 'mean', 'std'}

# 2) 配色与条件映射
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}
duration_list = [2.48, 2.48, 3.98, 3.98, 4.48, 4.48, 4.98, 4.98]
dotN_list     = [   6,    11,    6,    11,    6,    11,    6,     11]

# 3) 先构造一个排序后的 label 列表
labels_sorted = sorted(
    range(8),
    key=lambda l: (duration_list[l], dotN_list[l])
)

# 4) 绘图
plt.figure(figsize=(8,5))
for label in labels_sorted:
    entry = label_z[label]
    time = entry['time']
    mean = entry['mean']
    std  = entry['std']

    dur  = duration_list[label]
    dotN = dotN_list[label]
    color = duration_to_color[dur]

    # dotN=6: dashed, higher alpha; dotN=11: solid, lower alpha
    if dotN == 6:
        linestyle = '--'
        alpha = 0.5
    else:
        linestyle = '-'
        alpha = 0.2

    # 绘制平均曲线
    plt.plot(time, mean,
             color=color,
             linestyle=linestyle,
             label=f'{dur}s, dotN={dotN}')

    # 绘制置信带
    plt.fill_between(time,
                     mean - std,
                     mean + std,
                     color=color,
                     alpha=alpha)

# 5) 美化
plt.xlabel("Time since Set (frames)")
plt.ylabel("Network output $z_t$")
plt.title("$z$ over time for each condition")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', frameon=False)
plt.tight_layout()
plt.show()



# %%
# 所有folds结果一起画，并画出error bar

import pickle
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# 配置区：根据实际情况修改以下内容
# ====================================================================

# 1. 折数及其对应的文件路径（这里只示例 5 折，你可以按需增删）
n_folds = 5
fold_paths = [
    f"outputs/5c/inference_EM_4c_fold{i}.pkl" for i in range(n_folds)
]
# 2. 每个条件（label）对应的 duration 和 dotN（和之前保持一致）
duration_list = [2.48, 2.48, 3.98, 3.98, 4.48, 4.48, 4.98, 4.98]
dotN_list     = [   6,    11,    6,    11,    6,    11,    6,     11]
# 3. 画图时用的颜色映射（duration → 颜色）
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',   # 深蓝
    4.48: 'green',
    4.98: 'gold'
}
# ====================================================================

# -------------------------------
# （第一步）对每个 Fold 分别计算每个 label 的平均 Tp 和平均 ts
# -------------------------------

# 用来保存 5 折中，每折的 “每个 label 的 mean_Tp、mean_ts”
# 存储结构：shape = (n_folds, n_labels)
n_labels = 8
all_folds_mean_Tp = np.zeros((n_folds, n_labels), dtype=float)
all_folds_mean_ts = np.zeros((n_folds, n_labels), dtype=float)

for fold_idx, fp in enumerate(fold_paths):
    # 1. 加载这一折的 inference 返回的结果
    with open(fp, "rb") as f:
        data = pickle.load(f)
    # 假设 data['inference'] 是一个列表，每个元素都是 dict，至少包含 "Tp","ts","label"
    results = data['inference']

    # 2. 将这一折中，每个 label 下的 Tp 和 ts 分别收集到列表
    label_Tp = {l: [] for l in range(n_labels)}
    label_ts = {l: [] for l in range(n_labels)}
    for entry in results:
        Tp    = entry["Tp"]
        ts    = entry["ts"]
        label = entry["label"]
        # 这里如果 Tp 为 None，就跳过（表示模型没做出预测或无效）
        if Tp is None:
            continue
        label_Tp[label].append(Tp)
        label_ts[label].append(ts)

    # 3. 计算这一折中，每个 label 的平均 Tp 和平均 ts
    for l in range(n_labels):
        if len(label_Tp[l]) == 0:
            all_folds_mean_Tp[fold_idx, l] = np.nan
            all_folds_mean_ts[fold_idx, l] = np.nan
        else:
            all_folds_mean_Tp[fold_idx, l] = np.mean(label_Tp[l])
            all_folds_mean_ts[fold_idx, l] = np.mean(label_ts[l])

# 到这里为止：
# all_folds_mean_Tp.shape == (5, 8)
# all_folds_mean_ts.shape == (5, 8)
# 第 i 行对应 fold_i 下 8 个 label 的平均 Tp/ts


# -------------------------------
# （第二步）计算跨折平均和标准误（SEM）
# -------------------------------

# 先计算跨折的均值：shape = (n_labels,)
mean_Tp_across = np.nanmean(all_folds_mean_Tp, axis=0)  # 平均 Tp
mean_ts_across = np.nanmean(all_folds_mean_ts, axis=0)  # 平均 ts

# 计算标准误：SEM = 标准差 / sqrt(n_fold)，忽略 NaN
# np.nanstd 默认按样本总体标准差计算，这里为了近似用 ddof=0 即总体标准差
sem_Tp_across = np.nanstd(all_folds_mean_Tp, axis=0, ddof=0) / np.sqrt(n_folds)
sem_ts_across = np.nanstd(all_folds_mean_ts, axis=0, ddof=0) / np.sqrt(n_folds)

# 现在：
# mean_Tp_across.shape == (8,), 表示 8 个条件跨折的平均 Tp
# sem_Tp_across.shape  == (8,), 表示 8 个条件 Tp 的标准误
# mean_ts_across 和 sem_ts_across 同理


# -------------------------------
# （第三步）绘制带有 “1 SE 误差条” 的散点图
# -------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

for l in range(n_labels):
    x_mean = mean_ts_across[l]
    y_mean = mean_Tp_across[l]
    if np.isnan(x_mean) or np.isnan(y_mean):
        # 此条件可能在所有 fold 中都没有有效数据，跳过
        continue

    dur  = duration_list[l]
    dotN = dotN_list[l]
    color = duration_to_color[dur]
    # dotN=6 画得更饱满，dotN=11 透明度更低
    alpha = 0.6 if dotN == 6 else 0.3

    # 横向误差条（xerr）用 sem_ts_across[l]
    # 纵向误差条（yerr）用 sem_Tp_across[l]
    ax.errorbar(
        x_mean, y_mean,
        xerr=sem_ts_across[l],
        yerr=sem_Tp_across[l],
        fmt='o',                  # 点的形状为圆点
        ecolor=color,             # 误差线颜色与散点同色
        color=color,              # 散点颜色
        alpha=alpha,
        markersize=8,
        capsize=3,                # 误差条末端的小横线长度
        label=f"{dur}s, dotN={dotN}"
    )

# 画一条 45° 参考线 y=x
# 取所有跨折平均后 ts 与 Tp 的极值范围
valid_x = mean_ts_across[~np.isnan(mean_ts_across)]
valid_y = mean_Tp_across[~np.isnan(mean_Tp_across)]
mn = min(valid_x.min(), valid_y.min())
mx = max(valid_x.max(), valid_y.max())
ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)

# 坐标轴设置为等宽等高（正方形坐标系）
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("Trained interval (frames) 平均 ts")
ax.set_ylabel("Tp (frames) 平均 Tp")
ax.set_title("跨折平均: Tp vs Trained interval (带 1 SE 误差条)")

# 把图例放到右侧并缩小字体
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize='small',
    markerscale=0.8,
    frameon=False
)

plt.tight_layout()
plt.show()




















# %%
# dot图 合并相同duration
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 读取你的保存数据
with open("outputs/inference_DP_fold3_A_3.pkl", "rb") as f:
    data = pickle.load(f)
results = data['inference']

# with open("outputs/inference_DP_fold3_A_3.pkl", "rb") as f:
#     results = pickle.load(f)

# 1) 按 label 收集 Tp 和 ts
label_Tp = {l: [] for l in range(8)}
label_ts = {l: [] for l in range(8)}
for entry in results:
    Tp    = entry["Tp"]
    ts    = entry["ts"]
    label = entry["label"]
    if Tp is None: continue
    label_Tp[label].append(Tp)
    label_ts[label].append(ts)

# 2) 计算每个 label 的平均 Tp 和 ts
mean_Tp = []
mean_ts = []
for l in range(8):
    if not label_Tp[l]:
        mean_Tp.append(np.nan)
        mean_ts.append(np.nan)
    else:
        mean_Tp.append(np.mean(label_Tp[l]))
        mean_ts.append(np.mean(label_ts[l]))

# 配置
duration_list = [2.48,2.48,3.98,3.98,4.48,4.48,4.98,4.98]
dotN_list     = [   6,   11,   6,   11,   6,   11,   6,    11]
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}
markers = {6:'o', 11:'s'}

fig, ax = plt.subplots(figsize=(6,6))

# 3) 画 individual 条件点，并用线连 dot6->dot11
for dur in sorted(set(duration_list)):
    # 找到对应 index
    idxs = [i for i,d in enumerate(duration_list) if d==dur]
    xs = [mean_ts[i] for i in idxs]
    ys = [mean_Tp[i] for i in idxs]
    # 画这两点
    for i in idxs:
        d = duration_list[i]
        n = dotN_list[i]
        ax.scatter(mean_ts[i], mean_Tp[i],
                   color=duration_to_color[d],
                   marker=markers[n],
                   s=80, edgecolors='none',
                   label=f'{d}s, dotN={n}')
    # 用线连 dot6->dot11
    # 确保 order 6 then 11
    order = np.argsort(dotN_list[i] for i in idxs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    ax.plot(xs, ys, color=duration_to_color[dur], linestyle='-', lw=1.5, alpha=0.7)

# 4) 计算 aggregated averages
aggr_x = []
aggr_y = []
aggr_durs = sorted(set(duration_list))
for dur in aggr_durs:
    idxs = [i for i,d in enumerate(duration_list) if d==dur]
    aggr_x.append(np.nanmean([mean_ts[i] for i in idxs]))
    aggr_y.append(np.nanmean([mean_Tp[i] for i in idxs]))

# 5) 画 aggregated 点，用不同形状并连接
ax.scatter(aggr_x, aggr_y,
           color=[duration_to_color[d] for d in aggr_durs],
           marker='D', s=100, edgecolors='black',
           label='avg per duration')
ax.plot(aggr_x, aggr_y, color='black', linestyle='--', lw=1.5, label='duration avg trend')

# 45° 参考线
mn = min(min(aggr_x), min(aggr_y))
mx = max(max(aggr_x), max(aggr_y))
ax.plot([mn,mx], [mn,mx], 'k--')

# 设置等宽
ax.set_aspect('equal', 'box')
ax.set_xlabel("Trained interval (frames)")
ax.set_ylabel("Tp (frames)")
ax.set_title("Tp vs Trained interval")

# 去重图例
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          bbox_to_anchor=(1.05,1), loc='upper left',
          fontsize='small', frameon=False)

plt.tight_layout()
plt.show()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# 重新定义（或 import）你的 generate_ramp_target
def generate_ramp_target(
    ts_list: list[float],
    set_idx_list: list[int],
    seq_len_list: list[int],
    dotN_list,
    A: float = 3.0,
    alpha: float = 2.8,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    ramp_list = []
    for ts, set_idx, seq_len, dotN in zip(ts_list, set_idx_list, seq_len_list, dotN_list):
        target = torch.zeros(seq_len, device=device)
        # 从 Set 之后第一帧开始 ramp
        t_rel = torch.arange(1, seq_len - set_idx, device=device).float()
        ramp = A * (torch.exp(t_rel / (alpha * ts)) - 1)
        target[set_idx + 1 : set_idx + 1 + ramp.size(0)] = ramp
        ramp_list.append(target)
    return pad_sequence(ramp_list, batch_first=True)

# ====== 读取你训练用的数据集 (.pt) ======
dataset_path = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt"
dataset = torch.load(dataset_path)

# 提取每个 trial 的参数
ts_list = [s["ts"] for s in dataset]
set_idx_list = [s["set_idx"] for s in dataset]
seq_len_list = [s["seq_len"] for s in dataset]
dotN_list = [s["dotN"] for s in dataset]
label_list = [s["label"] for s in dataset]

# 如果你有 duration label 映射：
label_to_duration = {
    0: 2.48, 1: 2.48,
    2: 3.98, 3: 3.98,
    4: 4.48, 5: 4.48,
    6: 4.98, 7: 4.98
}
duration_list = [label_to_duration[l] for l in label_list]

# ====== 映射颜色和透明度 ======
duration_to_color = {
    2.48: 'purple',
    3.98: 'navy',
    4.48: 'green',
    4.98: 'gold'
}

# ====== 生成 ramp target ======
targets = generate_ramp_target(ts_list, set_idx_list, seq_len_list, dotN_list)

# ====== 画图（抽样每个 label 一个） ======
plotted = set()
plt.figure(figsize=(8, 5))
for i, (label, dur, dotN) in enumerate(zip(label_list, duration_list, dotN_list)):
    if label in plotted:
        continue
    set_idx = set_idx_list[i]
    y = targets[i, set_idx + 1 : seq_len_list[i]].cpu().numpy()
    x = np.arange(len(y))
    color = duration_to_color[dur]
    alpha = 0.6 if dotN == 6 else 0.3
    plt.plot(x, y, color=color, alpha=alpha, linewidth=2,
             label=f'{dur}s, dotN={dotN}')
    plotted.add(label)

# 添加阈值参考线
A, alpha_ramp = 3.0, 2.8
threshold = A * (np.exp(1 / alpha_ramp) - 1)
plt.axhline(threshold, linestyle='--', color='black', linewidth=1)

plt.xlabel("Time since Set (frames)")
plt.ylabel("Target ramp")
plt.title("Target ramp for each condition")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', frameon=False)
plt.tight_layout()
plt.show()

# %%
