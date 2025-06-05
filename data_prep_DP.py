# %%
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_DP.pt"

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
    if not filename.endswith(".csv"):
        continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])  # already in frames

        # 1) 构造 u，增加到 4 列： [ts_mask, dot_mask, x_coord, y_coord]
        u = torch.zeros(seq_len, 4)

        # 第一列：ts mask
        u[:ts, 0] = 1

        # 解析 loc1 字符串，得到每个 dot 的 (x,y) 坐标列表
        coords = ast.literal_eval(row['loc1'])
        if len(coords) != dotN:
            print(f"Warning: in {filename}, row {idx}, dotN={dotN} but coords length={len(coords)}")

        # 第二列：dot presentation mask；同时第三/四列填坐标
        pointer = 0
        for i in range(dotN):
            endp = pointer + dot_each
            if endp > seq_len:
                print(f"Warning: dot presentation exceeds seq_len in {filename}, row {idx}. pointer={pointer}, endp={endp}, seq_len={seq_len}")
                endp = seq_len

            # dot 出现时段
            u[pointer:endp, 1] = 1

            # 取对应坐标
            try:
                x, y = coords[i]
            except Exception:
                x, y = np.nan, np.nan
            u[pointer:endp, 2] = x
            u[pointer:endp, 3] = y

            # 移动指针
            pointer = endp
            if i < dotN - 1:
                pointer += isi

        # 2) 生成 label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: duration_1={duration_1}, dotN={dotN}")
            continue
        label = label_mapping[key]

        # 3) 保存 sample
        sample = {
            "u": u,                       # Tensor(seq_len, 4)
            "ts": ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label": label,
            "dotN": dotN
        }
        dataset.append(sample)

# 存为 .pt
torch.save(dataset, output_file)
print(f"Dataset saved successfully with a total of {len(dataset)} samples!")



# %%
# 归一化坐标
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_zcoreDP.pt"

dot_each = 9
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

# —— 第一遍：收集所有 x,y 值，计算全局 min/max —— #
all_x = []
all_y = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for s in df['loc1']:
        coords = ast.literal_eval(s)
        for (x,y) in coords:
            all_x.append(x); all_y.append(y)

x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
print(f"Global x range: [{x_min}, {x_max}], y range: [{y_min}, {y_max}]")

# 防止除0
if x_max == x_min: x_max += 1
if y_max == y_min: y_max += 1

# —— 第二遍：构造 dataset with normalized coords —— #
dataset = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])

        # u: [ts_mask, dot_mask, x_norm, y_norm]
        u = torch.zeros(seq_len, 4)
        u[:ts,0] = 1

        coords = ast.literal_eval(row['loc1'])
        pointer = 0
        for i in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            u[pointer:endp,1] = 1

            # 归一化到 [-1,1]
            x, y = coords[i]
            x_n = 2*(x - x_min)/(x_max - x_min) - 1
            y_n = 2*(y - y_min)/(y_max - y_min) - 1
            u[pointer:endp,2] = x_n
            u[pointer:endp,3] = y_n

            pointer = endp
            if i < dotN-1:
                pointer += isi

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! no label for {fn}, row {idx}")
            continue
        label = label_mapping[key]

        dataset.append({
            "u":        u,
            "ts":       ts,
            "set_idx":  set_idx,
            "seq_len":  seq_len,
            "label":    label,
            "dotN":     dotN
        })

torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")
# %%
# z-score
import os
import pandas as pd
import numpy as np
import torch
import ast

input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_zscoreDP_1stFrame.pt"

dot_each = 9

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

#######################
#### 1️⃣ 先统计所有坐标
#######################

all_x = []
all_y = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, filename))
    for loc_str in df['loc1']:
        coords = ast.literal_eval(loc_str)
        for x, y in coords:
            all_x.append(x)
            all_y.append(y)

all_x = np.array(all_x)
all_y = np.array(all_y)

x_mean = np.mean(all_x)
x_std  = np.std(all_x)
y_mean = np.mean(all_y)
y_std  = np.std(all_y)

print(f"x 坐标均值={x_mean:.2f}，标准差={x_std:.2f}")
print(f"y 坐标均值={y_mean:.2f}，标准差={y_std:.2f}")

#######################
#### 2️⃣ 构造 dataset
#######################

dataset = []
location_scale = 0.1   # 控制位置变量的缩放（建议从0.1开始）

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])  # 已是 frame

        u = torch.zeros(seq_len, 4)
        u[:ts, 0] = 1  # ts mask

        coords = ast.literal_eval(row['loc1'])
        if len(coords) != dotN:
            print(f"Warning: {filename}, row {idx}, dotN={dotN} but coords len={len(coords)}")

        # pointer = 0
        # for i in range(dotN):
        #     endp = pointer + dot_each
        #     if endp > seq_len:
        #         print(f"Warning: {filename}, row {idx}, dot超过seq_len")
        #         endp = seq_len

        #     u[pointer:endp, 1] = 1  # dot mask

        #     # 标准化坐标（z-score）并缩放
        #     x, y = coords[i]
        #     x_norm = ((x - x_mean) / x_std) * location_scale
        #     y_norm = ((y - y_mean) / y_std) * location_scale

        #     u[pointer:endp, 2] = x_norm
        #     u[pointer:endp, 3] = y_norm

        #     pointer = endp
        #     if i < dotN - 1:
        #         pointer += isi

        pointer = 0
        for i in range(dotN):
            endp = min(pointer + dot_each, seq_len)

            # dot mask：整个持续期都为 1
            u[pointer:endp, 1] = 1

            # 只在第一帧写坐标
            x, y = coords[i]
            x_norm = ((x - x_mean) / x_std) * location_scale
            y_norm = ((y - y_mean) / y_std) * location_scale
            u[pointer, 2] = x_norm
            u[pointer, 3] = y_norm
            # 剩余帧数坐标通道都保持 0（已初始化为 0）

            # 移动到下一个 dot
            pointer = endp
            if i < dotN - 1:
                pointer += isi

        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: duration_1={duration_1}, dotN={dotN}")
            continue
        label = label_mapping[key]

        sample = {
            "u": u,
            "ts": ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label": label,
            "dotN": dotN
        }
        dataset.append(sample)

#######################
#### 3️⃣ 保存
#######################

torch.save(dataset, output_file)
print(f"Dataset saved! Total samples = {len(dataset)}")





# %%
# 加一列代表不同（位置）的点
import os
import pandas as pd
import numpy as np
import torch

# set up input and output folders
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_DP_id.pt"

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

dataset = []

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])  # in frames

        # now u has 3 columns: [ts_mask, dot_mask, dot_id]
        u = torch.zeros(seq_len, 3)
        # 1) ts mask
        u[:ts, 0] = 1

        # 2) dot mask + dot id
        pointer = 0
        for dot_idx in range(dotN):
            endp = pointer + dot_each
            if endp > seq_len:
                print(f"Warning: dot exceeds seq_len in {filename}, row {idx}")
                endp = seq_len
            # mark dot presence
            u[pointer:endp, 1] = 1
            # mark which dot (1-based index)
            u[pointer:endp, 2] = dot_idx + 1

            pointer = endp
            if dot_idx < dotN - 1:
                pointer += isi

        # 3) label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label mapping for {filename}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        sample = {
            "u":       u,         # Tensor(seq_len, 3)
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        }
        dataset.append(sample)

torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples!")

# %%

# %%
# 加一列 但是是欧式距离
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_delta_rDP.pt"

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

# —— 第一遍：收集所有 delta_r —— #
all_dr = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for loc_str in df['loc1']:
        coords = ast.literal_eval(loc_str)
        prev_x, prev_y = None, None
        for x, y in coords:
            if prev_x is None:
                dr = 0.0
            else:
                dr = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            all_dr.append(dr)
            prev_x, prev_y = x, y

all_dr = np.array(all_dr)
dr_mean = all_dr.mean()
dr_std  = all_dr.std()
# 防止除以零
dr_std = dr_std if dr_std != 0 else 1.0
print(f"delta_r mean={dr_mean:.2f}, std={dr_std:.2f}")

# 缩放因子，可根据实验调整
delta_scale = 0.1

# —— 第二遍：构造 dataset —— #
dataset = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])

        # u: [ts_mask, dot_mask, delta_r]
        u = torch.zeros(seq_len, 3)
        u[:ts, 0] = 1

        coords = ast.literal_eval(row['loc1'])
        prev_x, prev_y = None, None
        pointer = 0
        for x, y in coords:
            endp = min(pointer + dot_each, seq_len)
            # dot mask
            u[pointer:endp, 1] = 1

            # 计算 delta_r 并归一化
            if prev_x is None:
                dr = 0.0
            else:
                dr = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            prev_x, prev_y = x, y
            dr_n = ((dr - dr_mean) / dr_std) * delta_scale
            u[pointer:endp, 2] = dr_n

            pointer = endp
            if len(coords) > 1 and pointer < seq_len and coords.index((x,y)) < len(coords)-1:
                pointer += isi

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label for {fn}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        dataset.append({
            "u":       u,
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        })

# 保存
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")

# %%
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_inhibitoryDP.pt"

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

# ———— 第一步：统计所有 delta_r ———#
all_dr = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for loc_str in df['loc1']:
        coords = ast.literal_eval(loc_str)
        prev = None
        for x, y in coords:
            if prev is None:
                dr = 0.0
            else:
                px, py = prev
                dr = np.hypot(x - px, y - py)
            all_dr.append(dr)
            prev = (x, y)

all_dr = np.array(all_dr)
dr_mean = all_dr.mean()
dr_std  = all_dr.std() if all_dr.std() != 0 else 1.0
print(f"delta_r mean={dr_mean:.2f}, std={dr_std:.2f}")

delta_scale = 0.1  # 缩放系数，可调

# ———— 第二步：构造 dataset ———#
dataset = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])

        # u: [ts_mask, inhibitory_dot_mask, delta_r]
        u = torch.zeros(seq_len, 3)
        u[:ts, 0] = 1.0            # ts mask

        coords = ast.literal_eval(row['loc1'])
        prev = None
        pointer = 0

        for x, y in coords:
            endp = min(pointer + dot_each, seq_len)
            # 将 dot mask 设为 -1 表示抑制
            u[pointer:endp, 1] = -1.0

            # 计算 delta_r 并归一化
            if prev is None:
                dr = 0.0
            else:
                px, py = prev
                dr = np.hypot(x - px, y - py)
            prev = (x, y)

            dr_n = ((dr - dr_mean) / dr_std) * delta_scale
            u[pointer:endp, 2] = dr_n

            pointer = endp
            if pointer < seq_len:
                pointer += isi

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label for {fn}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        dataset.append({
            "u":       u, 
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        })

# 保存
torch.save(dataset, output_file)
print(f"Dataset saved successfully with {len(dataset)} samples to {output_file}")



# %%
# 加两列，欧式距离+dot出现的累积信息
import os
import ast
import pandas as pd
import numpy as np
import torch

# 输入/输出路径
input_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
output_file  = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_delta_r_cumDP.pt"

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

# —— 第一步：统计所有 delta_r —— #
all_dr = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for loc_str in df['loc1']:
        coords = ast.literal_eval(loc_str)
        prev_x, prev_y = None, None
        for x, y in coords:
            if prev_x is None:
                dr = 0.0
            else:
                dr = np.hypot(x - prev_x, y - prev_y)
            all_dr.append(dr)
            prev_x, prev_y = x, y

all_dr = np.array(all_dr)
dr_mean = all_dr.mean()
dr_std  = all_dr.std() if all_dr.std() != 0 else 1.0
print(f"delta_r mean={dr_mean:.2f}, std={dr_std:.2f}")

delta_scale = 0.1  # 缩放系数，可调

# —— 第二步：构造 dataset —— #
dataset = []
for fn in os.listdir(input_folder):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(input_folder, fn))
    for idx, row in df.iterrows():
        ts      = int(row['ts'])
        set_idx = int(row['set_idx'])
        seq_len = int(row['seq_len'])
        dotN    = int(row['dotN_1'])
        isi     = int(row['isi_1'])

        # 初始 u: [ts_mask, dot_mask, delta_r]
        u = torch.zeros(seq_len, 3)
        u[:ts, 0] = 1.0  # ts mask

        coords = ast.literal_eval(row['loc1'])
        prev_x, prev_y = None, None
        pointer = 0
        for x, y in coords:
            endp = min(pointer + dot_each, seq_len)
            # dot mask
            u[pointer:endp, 1] = 1.0
            # 计算 delta_r 并归一化
            if prev_x is None:
                dr = 0.0
            else:
                dr = np.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
            dr_n = ((dr - dr_mean) / dr_std) * delta_scale
            u[pointer:endp, 2] = dr_n
            pointer = endp
            if pointer < seq_len and len(coords) > 1 and coords.index((x,y)) < len(coords)-1:
                pointer += isi

        # —— 新增 cum_dots 通道 —— #
        # 计算到当前帧为止累计 dot 数，并归一化
        cum_dots = torch.cumsum(u[:,1], dim=0)
        cum_dots = cum_dots / cum_dots[-1].clamp(min=1)
        # 拼接为第四列
        u = torch.cat([u, cum_dots.unsqueeze(1)], dim=1)  # shape: (seq_len, 4)

        # label
        duration_1 = round(float(row['duration_1']), 2)
        key = (duration_1, dotN)
        if key not in label_mapping:
            print(f"Warning! No label for {fn}, row {idx}: {key}")
            continue
        label = label_mapping[key]

        dataset.append({
            "u":       u,
            "ts":      ts,
            "set_idx": set_idx,
            "seq_len": seq_len,
            "label":   label,
            "dotN":    dotN
        })

# 保存 dataset
torch.save(dataset, output_file)
print(f"Saved {len(dataset)} samples to {output_file}")














# %%
# check data

import torch

# 1. 加载整个 dataset
dataset = torch.load("data/dataset_delta_r_cumDP.pt")  # 返回的是一个 list

# 2. 查看 dataset 大小
print(f"Total samples: {len(dataset)}")

# 3. 取出第 0 条样例（也可以换成任何索引）
sample = dataset[1]

# 4. 查看这个样例包含哪些字段
print("Sample keys:", sample.keys())

# 5. 查看 u 的形状和前几行内容
u = sample["u"]  # Tensor(seq_len, 4)
print("u shape:", u.shape)
print("u (first 10 rows):")
print(u[:100])
# %%
