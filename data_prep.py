# 数据处理流程
# 1. 检查数据中是否存在 NaN 和 inf
# 2. 把数据转换成帧数
# 3. rnn训练数据预处理
# 4. 查看训练数据

# %%
# 1. check NaN and inf
import os
import pandas as pd
import numpy as np

input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti"   # original data path

files_with_nan = []
files_with_inf = []

nan_counts = {}  # file name -> number of NaN rows
inf_counts = {}  # file name -> number of inf rows

total_samples = 0   # the total number of samples (rows) across all files
total_nan = 0       # the total number of rows containing NaN
total_inf = 0       # the total number of rows containing inf/-inf

# iterate through all csv files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        cols_to_check = ['duration_1', 'ITI_s', 'rep_time']

        total_samples += len(df)

        # check NaN
        nan_rows = df[cols_to_check].isnull().any(axis=1)
        nan_count = nan_rows.sum()

        if nan_count > 0:
            files_with_nan.append(filename)
            nan_counts[filename] = nan_count
            total_nan += nan_count
            print(f"{filename} — indices of rows containing NaN:", df.index[nan_rows].tolist())


        # check inf
        inf_rows = np.isinf(df[cols_to_check]).any(axis=1)
        inf_count = inf_rows.sum()

        if inf_count > 0:
            files_with_inf.append(filename)
            inf_counts[filename] = inf_count
            total_inf += inf_count
            print(f"{filename} — indices of rows containing inf:", df.index[inf_rows].tolist())


# print results
print("\n====== Files Summary ======")
if files_with_nan:
    print("The following file(s) contain NaN values：")
    for f in files_with_nan:
        print(f"{f} ：{nan_counts[f]} contain(s) NaN")
else:
    print("No NaN values were found in any of the files!")

print("\n------------------------")

if files_with_inf:
    print("The following file(s) contain inf or -inf values ：")
    for f in files_with_inf:
        print(f"{f} ：{inf_counts[f]} contain(s) inf/-inf")
else:
    print("No inf or -inf values were found in any of the files!")


print("\n====== Overall Summary ======")
print(f"Total number of samples (total rows across all CSV files): {total_samples}")
print(f"Total number of rows containing NaN: {total_nan}")
print(f"Total number of rows containing inf/-inf: {total_inf}")



# %%
# 2. ms -> frames
import os
import pandas as pd
import numpy as np

# set up input and output folders
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti"    # original data path
output_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"  # output data path
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Remove rows containing NaN or infinite values (important!)
        df = df.replace([np.inf, -np.inf], np.nan) # replace inf/-inf with NaN
        df = df.dropna(subset=['duration_1', 'ITI_s', 'rep_time'])

        # Add four new columns (convert to frames by multiplying by 60 and rounding)
        df['ts'] = (df['duration_1'] * 60).round().astype(int)
        df['iti'] = (df['ITI_s'] * 60).round().astype(int)
        df['seq_len'] = ((df['duration_1'] + df['ITI_s'] + df['rep_time']) * 60).round().astype(int)
        df['set_idx'] = ((df['duration_1'] + df['ITI_s']) * 60).round().astype(int) + 1  # duration_1 + ITI_s后一帧

        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)

print("All files have been processed!")



# %%
# 3. dataset for rnn
import os
import pandas as pd
import numpy as np
import torch

# set up input and output folders
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames" 
output_file = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt"

# the number of frames for each dot
dot_each = 9

# label mapping
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

# create an empty list to store the dataset
dataset = []

# iterate through all csv files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            ts = row['ts']        # frame
            set_idx = row['set_idx']
            seq_len = row['seq_len']
            dotN = int(row['dotN_1'])
            isi = row['isi_1']     # frame, doesn't need to convert

            # generate u
            u = torch.zeros(seq_len, 2)

            # First column: ts mask (stimulus duration mask)
            ts_frames = int(ts)
            u[:ts_frames, 0] = 1

            # Second column: dot presentation mask
            dot_times = dotN
            interval_times = dotN - 1
            total_dot_frames = dot_times * dot_each + interval_times * isi


            # Dot presentation over the timeline
            pointer = 0
            for _ in range(dot_times):
                end_pointer = pointer + dot_each
                if end_pointer > seq_len:
                    print(f"Warning: dot presentation exceeds seq_len in this trial. pointer={pointer}, end_pointer={end_pointer}, seq_len={seq_len}")
                    end_pointer = seq_len  # in case it exceeds seq_len
                u[pointer:end_pointer, 1] = 1  # dot presentation
                pointer += dot_each
                if _ < dot_times - 1:
                    pointer += isi  # add isi between dots

            # label
            duration_1 = round(float(row['duration_1']), 2)  # use duration_1 for label mapping
            key = (duration_1, dotN)
            if key not in label_mapping:
                print(f"Warning! No label mapping found in {filename}, row {idx}: duration_1={duration_1}, dotN={dotN}")
                continue
            label = label_mapping[key]

            # create a sample
            sample = {
                "u": u,
                "ts": int(ts),
                "set_idx": int(set_idx),
                "seq_len": int(seq_len),
                "label": int(label),
                "dotN": int(dotN)
            }
            dataset.append(sample)

# save as a PyTorch dataset
torch.save(dataset, output_file)
print(f"Dataset saved successfully with a total of {len(dataset)} samples!")

# %%
# 查看训练数据
import torch

# 加载数据
dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")
print(f"共有 {len(dataset)} 条样本")

# 打印前10个样本
for i, sample in enumerate(dataset[:10]):
    print(f"样本 {i}:")
    print("u.shape:", sample['u'].shape)
    print("ts:", sample['ts'])
    print("set_idx:", sample['set_idx'])
    print("seq_len:", sample['seq_len'])
    print("label:", sample['label'])
    print("dotN:", sample['dotN'])
    print("-------------------------")

# %%
# sanity check
import torch

# 加载 trials_list
dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")

mismatch_count = 0

for i, trial in enumerate(dataset):
    u_len = trial["u"].shape[0]
    seq_len = trial["seq_len"]  # trial 构造时保存的 seq_len
    if u_len != seq_len:
        print(f"Mismatch found in trial {i}: u.shape[0]={u_len}, trial['seq_len']={seq_len}")
        mismatch_count += 1

if mismatch_count == 0:
    print("All trials passed the check: u.shape[0] matches trial['seq_len'].")
else:
    print(f"Total mismatches found: {mismatch_count}")








# 开始尝试新的数据集训练
# %%
# +set开始序列
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_with_setmask.pt"

# 每个 dot 持续的帧数
dot_each = 9

# label 映射
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

dataset = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"): continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])        # stimulus duration in frames
        set_idx = int(row['set_idx'])   # production onset index
        seq_len = int(row['seq_len'])   # total frames in trial
        dotN    = int(row['dotN_1'])    # number of dots
        isi     = int(row['isi_1'])     # inter-stimulus interval in frames

        # u: [ts_mask, dot_mask, set_mask]
        u = torch.zeros(seq_len, 3)

        # 第一列：cue阶段掩码
        u[:ts, 0] = 1.0

        # 第二列：dot呈现掩码
        pointer = 0
        for _ in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp, 1] = 1.0
            pointer = endp
            if _ < dotN - 1:
                pointer += isi
                if pointer > seq_len:
                    pointer = seq_len

        # 第三列：Set脉冲掩码，仅在 production 开始那一帧置 1
        if 0 <= set_idx < seq_len:
            u[set_idx, 2] = 1.0
        else:
            print(f"Warning: set_idx {set_idx} out界于 seq_len={seq_len} in {filename}, row {idx}")

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,         # Tensor of shape (seq_len, 3)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        }
        dataset.append(sample)

# 保存为 PyTorch 格式
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples to {output_file}")




# %%
# +set开始序列（但是111110000）
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_with_setmask1.pt"

# 每个 dot 持续的帧数
dot_each = 9

# label 映射
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

dataset = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"): continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])        # stimulus duration in frames
        set_idx = int(row['set_idx'])   # production onset index
        seq_len = int(row['seq_len'])   # total frames in trial
        dotN    = int(row['dotN_1'])    # number of dots
        isi     = int(row['isi_1'])     # inter-stimulus interval in frames

        # u: [ts_mask, dot_mask, set_mask]
        u = torch.zeros(seq_len, 3)

        # 第一列：cue阶段掩码
        u[:ts, 0] = 1.0

        # 第二列：dot呈现掩码
        pointer = 0
        for _ in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp, 1] = 1.0
            pointer = endp
            if _ < dotN - 1:
                pointer += isi
                if pointer > seq_len:
                    pointer = seq_len

        # 第三列：Set阶段之前为1，从 set_idx 开始（含）全部为0
        # 设置前面时刻为1，set_idx及之后保持0
        if 0 <= set_idx <= seq_len:
            u[:set_idx, 2] = 1.0
        else:
            print(f"Warning: set_idx {set_idx} 一bound于 seq_len={seq_len} in {filename}, row {idx}")

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,         # Tensor of shape (seq_len, 3)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        }
        dataset.append(sample)

# 保存为 PyTorch 格式
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples to {output_file}")






# %%
# 3. dataset for rnn — 仅保留 ts mask
import os
import pandas as pd
import torch

# set up input and output paths
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_only.pt"

# label mapping
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

dataset = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"): 
        continue
    df = pd.read_csv(os.path.join(input_folder, filename))

    for idx, row in df.iterrows():
        ts = int(row['ts'])            # stimulus duration in frames
        set_idx = int(row['set_idx'])  # production onset index (unused here)
        seq_len = int(row['seq_len'])  # total frames in trial
        dotN = int(row['dotN_1'])

        # u: 只有 ts mask
        u = torch.zeros(seq_len, 1)
        u[:ts, 0] = 1.0

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label mapping for {filename}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            'u': u,                # Tensor(seq_len, 1)
            'ts': ts,
            'set_idx': set_idx,
            'seq_len': seq_len,
            'label': label,
            'dotN': dotN
        }
        dataset.append(sample)

# save dataset
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples with ts-only mask to {output_file}")


# %%
# ts_mask + set
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_set_mask.pt"

# label 映射
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

dataset = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"): continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])        # stimulus duration in frames
        set_idx = int(row['set_idx'])   # production onset index
        seq_len = int(row['seq_len'])   # total frames in trial
        dotN    = int(row['dotN_1'])    # number of dots
        # 我们删除 dot_mask，所以不再需要 isi 和 dot_each

        # u: [ts_mask, set_mask]
        u = torch.zeros(seq_len, 2)

        # 第一列：cue阶段掩码
        u[:ts, 0] = 1.0

        # 第二列：Set掩码，从 set_idx 开始为 1
        if 0 <= set_idx < seq_len:
            # 如果想让 set 这一帧及之后都为 1：
            u[set_idx:, 1] = 1.0
        else:
            print(f"Warning: set_idx {set_idx} 超出范围 seq_len={seq_len} in {filename}, row {idx}")

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,         # Tensor of shape (seq_len, 2)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        }
        dataset.append(sample)

# 保存为 PyTorch 格式
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples to {output_file}")





# # %%
# # 数据可视化
# import torch
# import matplotlib.pyplot as plt
# from collections import Counter

# dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")
# labels = [sample['label'] for sample in dataset]

# counter = Counter(labels)
# plt.bar(counter.keys(), counter.values())
# plt.xlabel("Label (Condition Index)")
# plt.ylabel("Number of Trials")
# plt.title("Trial Count per Condition")
# plt.xticks(range(8))
# plt.show()

# # %%
# import seaborn as sns
# import pandas as pd

# dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")

# data = {
#     "label": [],
#     "ts": [],
#     "set_idx": [],
#     "seq_len": [],
# }

# for sample in dataset:
#     data["label"].append(sample["label"])
#     data["ts"].append(sample["ts"])
#     data["set_idx"].append(sample["set_idx"])
#     data["seq_len"].append(sample["seq_len"])

# df = pd.DataFrame(data)

# plt.figure(figsize=(10, 6))
# sns.boxplot(x="label", y="ts", data=df)
# plt.title("Stimulus Duration (ts) by Label")
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x="label", y="set_idx", data=df)
# plt.title("Set Onset Index by Label")
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x="label", y="seq_len", data=df)
# plt.title("Trial Length (seq_len) by Label")
# plt.show()

# # %%
# df["prod_len"] = df["seq_len"] - df["set_idx"]
# df["prod_ratio"] = df["prod_len"] / df["ts"]

# plt.figure(figsize=(10,6))
# sns.boxplot(x="label", y="prod_ratio", data=df)
# plt.title("Production Phase Ratio by Label")
# plt.ylabel("Production / Ts")
# plt.show()

# # %%
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch

# # 颜色映射
# duration_to_color = {
#     2.48: 'purple',
#     3.98: 'navy',
#     4.48: 'green',
#     4.98: 'gold'
# }

# # 加载数据
# dataset = torch.load("/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt")

# # 提取信息
# data = {"label": [], "ts": [], "dotN": []}
# for sample in dataset:
#     data["label"].append(sample["label"])
#     data["ts"].append(sample["ts"])
#     data["dotN"].append(sample["dotN"])

# df = pd.DataFrame(data)

# # label 对应 duration 和 dotN（假设你按之前设定的顺序）
# label_to_duration = {
#     0: 2.48, 1: 2.48, 2: 3.98, 3: 3.98,
#     4: 4.48, 5: 4.48, 6: 4.98, 7: 4.98
# }
# df["duration"] = df["label"].map(label_to_duration)

# # 可视化
# plt.figure(figsize=(7, 5))
# for label in sorted(df["label"].unique()):
#     subset = df[df["label"] == label]
#     dur = label_to_duration[label]
#     dotN = subset["dotN"].iloc[0]
#     alpha = 0.7 if dotN == 6 else 0.3
#     color = duration_to_color[dur]
#     plt.scatter([label], [subset["ts"].mean()], color=color, alpha=alpha, s=100, label=f"{dur}s, dotN={dotN}")

# plt.xlabel("Label")
# plt.ylabel("Stimulus Duration (ts)")
# plt.title("ts per Label (Color: Duration, Alpha: dotN)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
# plt.tight_layout()
# plt.show()


# %%
import os
import pandas as pd
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_dotN.pt"

# label 映射 (duration, dotN) -> label index
label_mapping = {
    (2.48, 6): 0,
    (2.48, 11): 1,
    (3.98, 6): 2,
    (3.98, 11): 3,
    (4.48, 6): 4,
    (4.48, 11): 5,
    (4.98, 6): 6,
    (4.98, 11): 7,
}

dataset = []

for fn in os.listdir(input_folder):
    if not fn.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts = int(row['ts'])
        seq_len = int(row['seq_len'])
        dotN = int(row['dotN_1'])

        # 构造 u: 两列 [ts_mask, dotN_norm]
        u = torch.zeros(seq_len, 2)
        # 第一列：ts_mask
        u[:ts, 0] = 1.0
        # 第二列：dotN 注入，仅在 cue 阶段（前 ts 帧）赋值
        dotN_norm = dotN / 11.0  # 归一化到 [6/11,1]
        u[:ts, 1] = dotN_norm

        # label 映射
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fn} row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            'u': u,                 # Tensor(seq_len, 2)
            'ts': ts,
            'set_idx': int(row['set_idx']),
            'seq_len': seq_len,
            'label': label,
            'dotN': dotN
        }
        dataset.append(sample)

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")

# %%
import os
import pandas as pd
import torch

input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_dotN_full.pt"
# 如果 dotN 的最大值不是 11，请改成实际最大 dotN
max_dotN = 11  

# 和你之前一样的 label 映射
label_mapping = {
    (2.48,  6): 0,
    (2.48, 11): 1,
    (3.98,  6): 2,
    (3.98, 11): 3,
    (4.48,  6): 4,
    (4.48, 11): 5,
    (4.98,  6): 6,
    (4.98, 11): 7,
}

dataset = []

for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])        # 刺激时长，帧数
        set_idx = int(row['set_idx'])   # 复现开始的索引
        seq_len = int(row['seq_len'])   # 总帧数
        dotN    = int(row['dotN_1'])    # 点的个数
        # normalize dotN to [0,1]
        dotN_feat = dotN / float(max_dotN)

        # 构造 u：[seq_len, 2]
        #   col0 = ts_mask; col1 = dotN_feat (constant)
        u = torch.zeros(seq_len, 2)
        u[:ts, 0] = 1.0
        u[:, 1] = dotN_feat

        # label mapping
        dur = round(float(row['duration_1']), 2)
        key = (dur, dotN)
        if key not in label_mapping:
            print(f"[WARN] {fn} row {idx}: no label for {key}, skip")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,          # Tensor(seq_len, 2)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        }
        dataset.append(sample)

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")
# %%
import os
import pandas as pd
import torch

# ——— 配置 ———
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_then_dotN.pt"
max_dotN = 11  # 你的 dotN 最大值，用于归一化

# duration+dotN → label，你原来的映射
label_mapping = {
    (2.48, 6): 0,  (2.48, 11): 1,
    (3.98, 6): 2,  (3.98, 11): 3,
    (4.48, 6): 4,  (4.48, 11): 5,
    (4.98, 6): 6,  (4.98, 11): 7,
}

dataset = []

for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])        # cue 阶段帧数
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])

        # 归一化 dotN
        dotN_feat = dotN / float(max_dotN)

        # 构造 u: [seq_len, 2]
        u = torch.zeros(seq_len, 2)

        # 第一列：ts_mask
        u[:ts, 0] = 1.0

        # 第二列：从第 ts 帧开始才注入 dotN_feat
        # （如果你想只在那一帧注入，用 u[ts,1]=dotN_feat；
        #  如果想从 ts 帧到末尾都注入，用下面这行）
        u[ts:, 1] = dotN_feat

        # label
        dur = round(float(row['duration_1']), 2)
        key = (dur, dotN)
        if key not in label_mapping:
            print(f"[WARN] {fn} row {idx}: unknown key {key}")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,         # Tensor(seq_len, 2)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN,
        }
        dataset.append(sample)

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")

# %%
import os
import ast
import pandas as pd
import torch

# --- 配置 ---
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dynamic_dotN.pt"
# dotN 最大，用于归一化累积计数
max_dotN = 11
# 每个 dot 持续的帧数
# 如果 dot_each 是全局变量，也可在此处重新定义
# dot_each = 9

def load_dot_each():
    # 如果你有定义 dot_each 全局变量，可注释此函数
    return 9  # 每个 dot 占用 9 帧

dot_each = load_dot_each()

# duration+dotN → label
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []
for fn in os.listdir(input_folder):
    if not fn.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])

        # 创建 u: [ts_mask, dot_mask, cum_dot_prop]
        u = torch.zeros(seq_len, 3)
        # 1) ts_mask 前 ts 帧为 1
        u[:ts, 0] = 1.0

        # 2) dot_mask + 3) 累积 dotN 比例
        pointer = 0
        count = 0
        for d in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            # dot_mask
            u[pointer:endp, 1] = 1.0
            # 累积计数 (归一化到 [1/dotN ... dotN/dotN])
            count += 1
            u[pointer:endp, 2] = count / float(dotN)
            # 更新指针
            pointer = endp
            if d < dotN - 1:
                pointer = min(pointer + isi, seq_len)
        # 4) 将累积比例延续到试次末尾
        if pointer < seq_len:
            u[pointer:, 2] = count / float(dotN)

        # 标签映射
        dur = round(float(row['duration_1']), 2)
        key = (dur, dotN)
        if key not in label_mapping:
            print(f"[WARN] {fn} row {idx}: 无法找到标签 {key}")
            continue
        label = label_mapping[key]

        dataset.append({
            'u':       u,
            'ts':      ts,
            'set_idx': set_idx,
            'seq_len': seq_len,
            'label':   label,
            'dotN':    dotN
        })

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples to {output_file}")






# 从这里开始ts加入抖动，rnn输入的是tm
# %%
# import os
# import ast
# import pandas as pd
# import numpy as np
# import torch

# # ---------- 配置 ----------
# input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
# output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar.pt"
# dot_each        = 9    # 每个 dot 持续帧数
# weber_fraction  = 0.15 # scalar variability 的 Weber fraction，可调
# np.random.seed(42)
# torch.manual_seed(42)

# # label 映射
# label_mapping = {
#     (2.48, 6): 0, (2.48, 11): 1,
#     (3.98, 6): 2, (3.98, 11): 3,
#     (4.48, 6): 4, (4.48, 11): 5,
#     (4.98, 6): 6, (4.98, 11): 7,
# }

# dataset = []

# for fname in os.listdir(input_folder):
#     if not fname.endswith(".csv"):
#         continue
#     df = pd.read_csv(os.path.join(input_folder, fname))
#     for idx, row in df.iterrows():
#         # ---- 原始 trial 信息 ----
#         ts       = int(row['ts'])        # 真正的刺激时长（帧）
#         seq_len  = int(row['seq_len'])   # trial 总帧长
#         dotN     = int(row['dotN_1'])    # dot 数量
#         isi      = int(row['isi_1'])     # dot 之间的间隔（帧）

#         # ---- 采样 noisy tm 作为 set_idx ----
#         # tm ~ N(ts, (weber_fraction * ts)^2)
#         tm = int(np.round(
#             np.random.normal(loc=ts, scale=weber_fraction * ts)
#         ))
#         # 保证在合理范围 [1, seq_len-1]
#         tm = max(1, min(tm, seq_len-1))
#         set_idx_noisy = tm

#         # ---- 构造输入 u (seq_len x 2) ----
#         # 列 0: ts_mask, 列 1: dot_mask
#         u = torch.zeros(seq_len, 2, dtype=torch.float32)

#         # 1) ts_mask: 前 ts 帧为 1
#         u[:ts, 0] = 1.0

#         # 2) dot_mask: 从 trial 开始连续 dotN 个 dot_each+isi 周期
#         pointer = 0
#         for d in range(dotN):
#             endp = min(pointer + dot_each, seq_len)
#             u[pointer:endp, 1] = 1.0
#             pointer = endp
#             if d < dotN - 1:
#                 pointer = min(pointer + isi, seq_len)

#         # ---- label ----
#         duration_1 = round(float(row['duration_1']), 2)
#         key = (duration_1, dotN)
#         if key not in label_mapping:
#             print(f"Warning: no label for {fname} row {idx} key={key}")
#             continue
#         label = label_mapping[key]

#         # ---- 保存 sample ----
#         sample = {
#             "u":                   u,               # Tensor(seq_len, 2)
#             "ts_true":             ts,              # 真实 ts（用来做 target）
#             "set_idx_noisy":       set_idx_noisy,   # noisy tm，用来切分估计/生产阶段
#             "seq_len":             seq_len,
#             "label":               label,
#             "dotN":                dotN,
#         }
#         dataset.append(sample)

# # save
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# torch.save(dataset, output_file)
# print(f"Saved {len(dataset)} trials to {output_file}")








# %%
import os
import ast
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar1.pt"
dot_each        = 9    # 每个 dot 持续帧数
weber_fraction  = 0.15 # scalar variability 的 Weber fraction，可调
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 真正的刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始 production onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量
        isi           = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为 set_idx_noisy ----
        # tm ~ N(ts, (weber_fraction * ts)^2)
        tm = int(np.round(
            np.random.normal(loc=ts, scale=weber_fraction * ts)
        ))
        # 保证在合理范围 [1, seq_len-1]
        tm = max(1, min(tm, seq_len-1))
        set_idx_noisy = tm

        # ---- 构造输入 u (seq_len x 2) ----
        # 列 0: ts_mask, 列 1: dot_mask
        u = torch.zeros(seq_len, 2, dtype=torch.float32)

        # 1) ts_mask: 前 ts 帧为 1
        u[:tm, 0] = 1.0

        # 2) dot_mask: 从 trial 开始连续 dotN 个 dot_each+isi 周期
        pointer = 0
        for _ in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp, 1] = 1.0
            pointer = endp
            if _ < dotN - 1:
                pointer += isi
                if pointer > seq_len:
                    pointer = seq_len

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":                u,               # Tensor(seq_len, 2)
            "ts":          ts,              # 真实 ts（用来做 target）
            "set_idx":     set_idx_true,    # 原始 set_idx
            "set_idx_noisy":    set_idx_noisy,   # noisy tm，用来切分估计/生产阶段
            "seq_len":          seq_len,
            "label":            label,
            "dotN":             dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")



# %%
import os
import ast
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar1_set.pt"
dot_each        = 9    # 每个 dot 持续帧数
weber_fraction  = 0.15 # scalar variability 的 Weber fraction，可调
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 真正的刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始 production onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量
        isi           = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为 set_idx_noisy ----
        # tm ~ N(ts, (weber_fraction * ts)^2)
        tm = int(np.round(
            np.random.normal(loc=ts, scale=weber_fraction * ts)
        ))
        # 保证在合理范围 [1, seq_len-1]
        tm = max(1, min(tm, seq_len-1))
        set_idx_noisy = tm

        # ---- 构造输入 u (seq_len x 2) ----
        # 列 0: ts_mask, 列 1: dot_mask
        u = torch.zeros(seq_len, 3, dtype=torch.float32)

        # 1) ts_mask: 前 ts 帧为 1
        u[:tm, 0] = 1.0

        # 2) dot_mask: 从 trial 开始连续 dotN 个 dot_each+isi 周期
        pointer = 0
        for _ in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp, 1] = 1.0
            pointer = endp
            if _ < dotN - 1:
                pointer += isi
                if pointer > seq_len:
                    pointer = seq_len
        
        # 复现开始脉冲: 仅在原始 set_idx_true 那一帧为 1
        u[set_idx_true, 2] = 1.0

        

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":                u,               # Tensor(seq_len, 2)
            "ts":          ts,              # 真实 ts（用来做 target）
            "set_idx":     set_idx_true,    # 原始 set_idx
            "set_idx_noisy":    set_idx_noisy,   # noisy tm，用来切分估计/生产阶段
            "seq_len":          seq_len,
            "label":            label,
            "dotN":             dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")






# %%
import os
import ast
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar2.pt"
dot_each        = 9    # 每个 dot 持续帧数
weber_fraction  = 0.15 # scalar variability 的 Weber fraction，可调
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 真正的刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始 production onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量
        isi           = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为 set_idx_noisy ----
        tm = int(np.round(
            np.random.normal(loc=ts, scale=weber_fraction * ts)
        ))
        tm = max(1, min(tm, seq_len-1))
        set_idx_noisy = tm

        # ---- 构造输入 u (seq_len x 2) ----
        # 列 0: ts_mask, 列 1: dot_mask (累加索引，归一化)
        u = torch.zeros(seq_len, 2, dtype=torch.float32)

        # 1) ts_mask: 前 ts 帧为 1
        u[:ts, 0] = 1.0

        # 2) dot_mask: 第 d 个 dot 使用值 d+1
        pointer = 0
        for d in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            mask_val = float(d + 1)
            u[pointer:endp, 1] = mask_val
            pointer = endp
            if d < dotN - 1:
                pointer = min(pointer + isi, seq_len)

        # 归一化 dot_mask 到 [0, 1]
        if dotN > 0:
            u[:, 1] /= float(dotN)

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":                u,               # Tensor(seq_len, 2)
            "ts":               ts,              # 真实 ts（用来做 target）
            "set_idx":         set_idx_true,    # 原始 set_idx
            "set_idx_noisy":  set_idx_noisy,   # 带噪声的 tm
            "seq_len":          seq_len,
            "label":            label,
            "dotN":             dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")

# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_speed_cue.pt"
dot_each = 9    # 每个 dot 持续帧数
weber_fraction = 0.15  # scalar variability 的 Weber fraction，可调
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        ts = int(row['ts'])              # 真正的刺激时长（帧）
        set_idx_true = int(row['set_idx'])
        seq_len = int(row['seq_len'])    # trial 总帧长
        dotN = int(row['dotN_1'])        # dot 数量
        isi = int(row['isi_1'])          # dot 之间的间隔

        # ---- 采样 noisy set_idx ----
        tm = int(np.round(np.random.normal(loc=ts, scale=weber_fraction * ts)))
        tm = max(1, min(tm, seq_len - 1))
        set_idx_noisy = tm

        # ---- 构造输入 u (seq_len x 2) ----
        u = torch.zeros(seq_len, 2, dtype=torch.float32)

        # 第一列: ts_mask
        u[:ts, 0] = 1.0

        # 第二列: perceived speed cue，全程恒定
        if dotN > 0:
            perceived_speed = dotN / ts  # 每帧多少个点（越多越快）
        else:
            perceived_speed = 0.0
        u[:, 1] = perceived_speed

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u": u,
            "ts": ts,
            "set_idx": set_idx_true,
            "set_idx_noisy": set_idx_noisy,
            "seq_len": seq_len,
            "label": label,
            "dotN": dotN,
        }
        dataset.append(sample)

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
# %%
# 5.10
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始生产 onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量
        isi           = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))

        # ---- 构造输入 u (seq_len x 2) ----
        u = torch.zeros(seq_len, 2, dtype=torch.float32)
        # 1) ts_mask: 前 tm_noisy 帧为 1 （加入抖动）
        u[:tm_noisy, 0] = 1.0
        # 2) dot_mask: 连续 dotN 个 dot_each+isi 周期
        pointer = 0
        for d in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp, 1] = 1.0
            pointer = endp
            if d < dotN - 1:
                pointer = min(pointer + isi, seq_len)

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample，保留原始 set_idx，并新增 set_idx_noisy ----
        sample = {
            "u":                 u,             # Tensor(seq_len, 2)
            "ts":                ts,            # 真实 ts（目标仍基于 ts）
            "set_idx":           set_idx_true,  # 原始生产 onset
            "set_idx_noisy":     tm_noisy,      # 抖动后的生产 onset
            "seq_len":           seq_len,
            "label":             label,
            "dotN":              dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")

# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar_set.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))

        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dot速度感知; 列2: 复现开始脉冲
        u = torch.zeros(seq_len, 3, dtype=torch.float32)
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        # dot速度感知: 全程恒定 speed = dotN/ts
        perceived_speed = dotN / ts
        u[:, 1] = perceived_speed
        # 复现开始脉冲: 仅在原始 set_idx_true 那一帧为 1
        u[set_idx_true, 2] = 1.0

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")

# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar_density_set.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dot速度感知; 列2: 复现开始脉冲
        u = torch.zeros(seq_len, 3, dtype=torch.float32)
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        # dot速度感知: 全程恒定 speed = dotN/ts
        dot_duration = dotN * dot_each
        gap_duration = (dotN - 1) * isi
        stim_duration = dot_duration + gap_duration  # 真实 dot 时序的总时间长度

        perceived_density = dotN / stim_duration
        u[:, 1] = perceived_density
        # 复现开始脉冲: 仅在原始 set_idx_true 那一帧为 1
        u[set_idx_true, 2] = 1.0

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")
# %%
# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_isi_set.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

max_isi = 49  # 最大 isi（帧数），用于归一化 dotN

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dot速度感知; 列2: 复现开始脉冲
        u = torch.zeros(seq_len, 3, dtype=torch.float32)
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = isi / max_isi
        # 复现开始脉冲: 仅在原始 set_idx_true 那一帧为 1
        u[set_idx_true, 2] = 1.0

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")


# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar_number_set.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dot速度感知; 列2: 复现开始脉冲
        u = torch.zeros(seq_len, 3, dtype=torch.float32)
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = dotN / 11
        # 复现开始脉冲: 仅在原始 set_idx_true 那一帧为 1
        u[set_idx_true, 2] = 1.0

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")
# %%
# 改成set从那一帧之后都是1

import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_scalar_number_set1.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dotN种类; 列2: 复现开始脉冲
        u = torch.zeros(seq_len, 3, dtype=torch.float32) 
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = dotN / 11
        # 复现开始脉冲: 从set_idx_true 那一帧开始到结束都为1
        u[set_idx_true:, 2] = 1.0

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")
# %%
# 尝试多个dotN特征
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN_features.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dotN种类; 列2: 复现开始脉冲; 列3: dotN速度感知
        u = torch.zeros(seq_len, 4, dtype=torch.float32) 
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = dotN / 11
        # 复现开始脉冲: 从set_idx_true 那一帧开始到结束都为1
        u[set_idx_true:, 2] = 1.0
        perceived_speed = dotN / ts
        u[:, 3] = perceived_speed

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")


# %%
# 5.14
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_tm_dotN.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dotN种类; 列2: 复现开始脉冲; 列3: dotN速度感知
        u = torch.zeros(seq_len, 2, dtype=torch.float32) 
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = dotN / 11

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")

# %%
# ts第一和最后为1，中间为0
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_ts_1_0_1_dotN.pt"
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始生产 onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量

        # ---- 构造输入 u (seq_len x 2) ----
        #   列 0: ts_mask（仅第 0 帧和第 ts-1 帧为 1）
        #   列 1: dotN 种类
        u = torch.zeros(seq_len, 2, dtype=torch.float32)
        # 刺激开始第 0 帧
        u[0, 0] = 1.0
        # 刺激结束第 ts-1 帧
        if ts - 1 < seq_len:
            u[ts - 1, 0] = 1.0
        # dotN 归一化 (0…1)
        u[:, 1] = dotN / 11

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 2)
            "ts":            ts,           # 真实 ts
            "set_idx":       set_idx_true, # 原始生产 onset index
            "label":         label,        # condition id
            "dotN":          dotN,         # dot 数量
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")
# %%
# %%
# ts第一和最后为1，中间为0，并且为抖动tm
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_tm_1_0_1_dotN.pt"
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts            = int(row['ts'])        # 刺激时长（帧）
        set_idx_true  = int(row['set_idx'])   # 原始生产 onset index
        seq_len       = int(row['seq_len'])   # trial 总帧长
        dotN          = int(row['dotN_1'])    # dot 数量

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))

        # ---- 构造输入 u (seq_len x 2) ----
        #   列 0: ts_mask（仅第 0 帧和第 ts-1 帧为 1）
        #   列 1: dotN 种类
        u = torch.zeros(seq_len, 2, dtype=torch.float32)
        # 刺激开始第 0 帧
        u[0, 0] = 1.0
        # 刺激结束第 ts-1 帧
        # u[:tm_noisy, 0] = 1.0
        if tm_noisy - 1 < seq_len:
            u[tm_noisy - 1, 0] = 1.0
        # dotN 归一化 (0…1)
        u[:, 1] = dotN / 11

        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 2)
            "ts":            ts,           # 真实 ts
            "set_idx":       set_idx_true, # 原始生产 onset index
            "set_idx_noisy": tm_noisy,    # 抖动后的生产 onset
            "label":         label,        # condition id
            "dotN":          dotN,         # dot 数量
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")



# %%
# 保存一个5列的数据
# 尝试多个dotN特征
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置 ----------
input_folder    = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_stim_5c.pt"
dot_each        = 9     # 每个 dot 持续帧数
weber_fraction  = 0.15  # scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

dataset = []

for fname in os.listdir(input_folder):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fname))
    for idx, row in df.iterrows():
        # ---- 原始 trial 信息 ----
        ts           = int(row['ts'])        # 刺激时长（帧）
        set_idx_true = int(row['set_idx'])   # 原始生产 onset index
        seq_len      = int(row['seq_len'])   # trial 总帧长
        dotN         = int(row['dotN_1'])    # dot 数量
        isi          = int(row['isi_1'])     # dot 之间的间隔（帧）

        # ---- 采样 noisy tm 作为抖动后的 set 时刻 ----
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = int(np.round(tm))
        tm_noisy = max(1, min(tm_noisy, seq_len - 1))


        # ---- 构造输入 u (seq_len x 3) ----
        # 列0: ts_mask（抖动）; 列1: dotN种类; 列2: 复现开始脉冲; 列3: dotN速度感知
        u = torch.zeros(seq_len, 5, dtype=torch.float32) 
        # ts_mask: 前 tm_noisy 帧为 1
        u[:tm_noisy, 0] = 1.0
        u[:, 1] = dotN / 11
        # 复现开始脉冲: 从set_idx_true 那一帧开始到结束都为1
        u[set_idx_true:, 2] = 1.0
        perceived_speed = dotN / ts
        u[:, 3] = perceived_speed

        dot_times = dotN
        interval_times = dotN - 1
        total_dot_frames = dot_times * dot_each + interval_times * isi
        
        # 列4：Dot presentation over the timeline
        pointer = 0
        for _ in range(dot_times):
            end_pointer = pointer + dot_each
            if end_pointer > seq_len:
                print(f"Warning: dot presentation exceeds seq_len in this trial. pointer={pointer}, end_pointer={end_pointer}, seq_len={seq_len}")
                end_pointer = seq_len  # in case it exceeds seq_len
                u[pointer:end_pointer, 4] = 1  # dot presentation
                pointer += dot_each
                if _ < dot_times - 1:
                    pointer += isi  # add isi between dots



        # ---- label ----
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning: no label for {fname} row {idx} key={key}")
            continue
        label = label_mapping[key]

        # ---- 保存 sample ----
        sample = {
            "u":             u,            # Tensor(seq_len, 3)
            "ts":            ts,           # 真实 ts（目标仍基于 ts）
            "set_idx":       set_idx_true, # 原始生产 onset
            "set_idx_noisy": tm_noisy,     # 抖动后的生产 onset
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
        }
        dataset.append(sample)

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} trials to {output_file}")
# %%
