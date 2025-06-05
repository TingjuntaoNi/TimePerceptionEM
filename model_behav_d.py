# %%
import torch
import numpy as np
from collections import defaultdict
import pandas as pd

# 加载数据
dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")  # 你本地的路径

# 计算每个 label 的平均 Tp（Tp = seq_len - set_idx）
tp_by_label = defaultdict(list)
for trial in dataset:
    label = trial["label"]
    tp = trial["seq_len"] - trial["set_idx"]
    tp_by_label[label].append(tp)

# 整理结果
summary = []
for label in sorted(tp_by_label.keys()):
    tps = tp_by_label[label]
    mean_tp = np.mean(tps)
    std_tp = np.std(tps)
    summary.append((label, mean_tp, std_tp))

# 输出 DataFrame
df = pd.DataFrame(summary, columns=["label", "mean_tp", "std_tp"])
print(df)


# %%
