# %%
import os
import pandas as pd
import numpy as np
import torch

# ---------- 配置区，请根据实际改 ----------
pre_folder      = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
resample_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/resample"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_EM.pt"

weber_fraction  = 0.15     # Scalar variability 的 Weber fraction
np.random.seed(42)
torch.manual_seed(42)

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

def parse_series(val, length, dtype=float):
    """
    把 '0,1,0,1,…' 或 '[0,1,0,1,…]' 解析成长度 length 的 list；
    如果是标量，扩成常数向量；否则返回全 0 向量。
    """
    if isinstance(val, str):
        try:
            s = val.strip('[]').split(',')
            return [dtype(x) for x in s]
        except:
            return [dtype(0)] * length
    elif np.isscalar(val):
        return [dtype(val)] * length
    else:
        return [dtype(0)] * length

dataset = []

for fname in os.listdir(pre_folder):
    if not fname.endswith(".csv"):
        continue

    path_pre = os.path.join(pre_folder, fname)
    path_res = os.path.join(resample_folder, fname)
    if not os.path.exists(path_res):
        print(f"⚠️ 找不到对应的 resample 文件：{path_res}")
        continue

    # 读取两个表格
    df_pre = pd.read_csv(path_pre)
    df_res = pd.read_csv(path_res)

    # —— 在 pre 表格中，按 block 累计计算全局 Trial_index —— #
    # 1. 统计每个 block 含有多少 trial
    block_counts = df_pre.groupby('block').size().sort_index()
    # 2. 生成每个 block 的起始 offset
    offset_map = {}
    cum = 0
    for blk in block_counts.index:
        offset_map[blk] = cum
        cum += block_counts[blk]
    # 3. 计算 Trial_index
    df_pre['Trial_index'] = df_pre.apply(
        lambda r: offset_map[r['block']] + int(r['trial']), axis=1
    )

    # —— 同理，在 resample 表格中也加 Trial_index（以防它也有 block/trial） —— #
    if 'block' in df_res.columns and 'trial' in df_res.columns:
        rc = df_res.groupby('block').size().sort_index()
        off_r = {}
        c = 0
        for b in rc.index:
            off_r[b] = c
            c += rc[b]
        df_res['Trial_index'] = df_res.apply(
            lambda r: off_r[r['block']] + int(r['trial']), axis=1
        )
    else:
        # 如果 resample 本来就有全局 trial_index 列，则改名统一
        if 'trial_index' in df_res.columns:
            df_res.rename(columns={'trial_index':'Trial_index'}, inplace=True)

    # —— 对每一行 pre 数据，构造 u 并匹配 resample —— #
    for _, row in df_pre.iterrows():
        ts           = int(row['ts'])
        seq_len      = int(row['seq_len'])
        dotN         = int(row['dotN_1'])
        set_idx_true = int(row['set_idx'])
        trial_idx    = int(row['Trial_index'])

        # 产生 noisy tm
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        tm_noisy = max(1, min(seq_len - 1, int(round(tm))))

        # 构造 u：(seq_len, 9)
        u = torch.zeros(seq_len, 9, dtype=torch.float32)
        u[:tm_noisy, 0] = 1.0           # ts_mask
        u[:, 1] = dotN / 11             # dotN 归一化
        u[set_idx_true:, 2] = 1.0       # reproduction start
        u[:, 3] = dotN / ts             # perceived speed

        # 从 resample 中找到同一 Trial_index
        sub = df_res[df_res['Trial_index'] == trial_idx]
        if sub.empty:
            print(f"⚠️ {fname} Trial_index={trial_idx} 在 resample 中没找到")
            continue
        cres = sub.iloc[0]

        # 解析 display_box（0/1 序列）
        display_box = parse_series(cres['display_box'], ts, dtype=int)
        # 解析五个眼动特征
        sac = parse_series(cres['has_saccade'], ts)
        fix = parse_series(cres['has_fixation'], ts)
        blk = parse_series(cres['has_blink'], ts)
        rpf = parse_series(cres['rect_within_parafovea'], ts)
        dpf = parse_series(cres['dot_within_parafovea'], ts)

        # 对齐到 seq_len：仅当 display_box[t]==1 时写入，否则保持 0
        for t in range(min(ts, seq_len)):
            if display_box[t] == 1:
                u[t, 4] = sac[t]
                u[t, 5] = fix[t]
                u[t, 6] = blk[t]
                u[t, 7] = rpf[t]
                u[t, 8] = dpf[t]

        # 构造 label
        key = (round(float(row['duration_1']), 2), dotN)
        if key not in label_mapping:
            print(f"⚠️ 无法为 {fname} Trial_index={trial_idx} 找到 label，key={key}")
            continue
        label = label_mapping[key]

        dataset.append({
            "u":             u,  
            "ts":            ts,
            "set_idx":       set_idx_true,
            "set_idx_noisy": tm_noisy,
            "seq_len":       seq_len,
            "label":         label,
            "dotN":          dotN,
            "Trial_index":   trial_idx,
        })

# 存盘
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(dataset, output_file)
print(f"✅ 共保存 {len(dataset)} 个 samples 到：{output_file}")





# %%
import os
import pandas as pd
import numpy as np
import torch

# 这是结果比较好的5c (1：方框是否呈现*是否在注视方框，
# 2：点是否呈现*是否在注视点，3：点数量（6/11,全部时间点），
# 4：是否正在复现（复现的时间点为1），5：点速度（dotN/ts, 全部时间点）)


# -------------------- 配置区，请根据实际改 --------------------
pre_folder      = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
resample_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/resample"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_EM_5c.pt"

weber_fraction = 0.15    # Weber fraction
dot_each       = 9       # 每个 dot 持续帧数（固定不变）
# --------------------------------------------------------------

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

# 1）先定义一个“小工具”函数：把 '0,1,0,…' 或 '[0,1,0,…]' 这种字符串解析成长度为 length 的列表
def parse_series(val, length, dtype=float):
    """
    - 如果 val 是形如 "[0,1,0,1]" 或 "0,1,0,1" 的 str，就拆分、转换成 list[dtype] 并返回。
    - 如果 val 是一个标量（np.isscalar）或无法解析，就直接返回 [dtype(val)]*length 或 [0]*length。
    """
    if isinstance(val, str):
        try:
            s = val.strip('[]').split(',')
            return [dtype(x) for x in s]
        except:
            return [dtype(0)] * length
    elif np.isscalar(val):
        return [dtype(val)] * length
    else:
        return [dtype(0)] * length

# 2）构建一个空列表，用来一次性把所有 trial 的最终结果都 append 进来
new_dataset = []

# 3）遍历 pre_folder 下的所有 CSV 文件
for fname in os.listdir(pre_folder):
    if not fname.endswith(".csv"):
        continue

    path_pre = os.path.join(pre_folder, fname)
    path_res = os.path.join(resample_folder, fname)
    if not os.path.exists(path_res):
        print(f"⚠️ 找不到对应的 resample 文件：{path_res}，跳过此文件。")
        continue

    # —— 3.1 读取 pre 和 resample 两张表格 —— #
    df_pre = pd.read_csv(path_pre)
    df_res = pd.read_csv(path_res)

    # —— 3.2 在 pre 表格中，根据 'block' 累计计算全局 Trial_index —— #
    block_counts = df_pre.groupby('block').size().sort_index()
    offset_map = {}   # 存 block → 当前 block 的起始 trial_idx 偏移量
    cum = 0
    for blk in block_counts.index:
        offset_map[blk] = cum
        cum += block_counts[blk]

    df_pre['Trial_index'] = df_pre.apply(
        lambda r: offset_map[r['block']] + int(r['trial']), axis=1
    )

    # —— 3.3 在 resample 表格中，也生成同样的全局 Trial_index 列 —— #
    if 'block' in df_res.columns and 'trial' in df_res.columns:
        rc = df_res.groupby('block').size().sort_index()
        off_r = {}
        c = 0
        for b in rc.index:
            off_r[b] = c
            c += rc[b]
        df_res['Trial_index'] = df_res.apply(
            lambda r: off_r[r['block']] + int(r['trial']), axis=1
        )
    else:
        # 如果 resample 原本就有 trial_index，就重命名为统一字段
        if 'trial_index' in df_res.columns:
            df_res.rename(columns={'trial_index':'Trial_index'}, inplace=True)
        else:
            # 万一两个都没有，那就只能按行索引对应了（不推荐）
            print(f"❌ Warning: {fname} 的 resample 表里没有 block/trial 也没有 trial_index！")

    # —— 3.4 遍历 pre 表格里的每一行 trial，构建新的 u_new —— #
    for _, row in df_pre.iterrows():
        ts         = int(row['ts'])            # 刺激时长（帧）
        seq_len    = int(row['seq_len'])       # 序列总长度
        dotN       = int(row['dotN_1'])        # dot 总数（6 or 11）
        set_idx    = int(row['set_idx'])       # reproduction 开始的索引
        trial_idx  = int(row['Trial_index'])
        duration_1 = float(row['duration_1'])  # 用来映射 label 的 key
        isi_frames = int(row['isi_1'])         # 该 trial 的 dot 间隔帧数（动态读取）

        # —— 3.4.1 生成 noisy_tm —— #
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        # 限制 tm 在 [1, seq_len-1] 之间的整数
        tm_noisy = max(1, min(seq_len - 1, int(round(tm))))

        # —— 3.4.2 找到该 trial 在 resample 表里的对应行 —— #
        sub = df_res[df_res['Trial_index'] == trial_idx]
        if sub.empty:
            print(f"⚠️ 在 resample 中没找到 Trial_index = {trial_idx}（文件 {fname}），跳过此 trial。")
            continue
        cres = sub.iloc[0]

        # —— 3.4.3 解析 resample 表里的几列眼动序列 —— #
        # parse_series 会返回一个长度为 ts 的 Python list，我们再转成 Tensor
        display_box = torch.tensor(parse_series(cres['display_box'], ts, dtype=int), dtype=torch.float32)
        sac          = torch.tensor(parse_series(cres['has_saccade'], ts), dtype=torch.float32)
        fix          = torch.tensor(parse_series(cres['has_fixation'], ts), dtype=torch.float32)
        blk          = torch.tensor(parse_series(cres['has_blink'], ts), dtype=torch.float32)
        rpf          = torch.tensor(parse_series(cres['rect_within_parafovea'], ts), dtype=torch.float32)
        dpf          = torch.tensor(parse_series(cres['dot_within_parafovea'], ts), dtype=torch.float32)

        # —— 3.4.4 先根据 tm_noisy 构造 ts_mask_tm —— #
        # ts_mask_tm: 对于 t < tm_noisy, 取 1；否则 0；长度是 seq_len
        ts_mask_tm = torch.zeros(seq_len, dtype=torch.float32)
        ts_cut     = min(tm_noisy, seq_len)  # 防止 tm_noisy 跑出范围
        ts_mask_tm[:ts_cut] = 1.0

        # —— 3.4.5 根据 dotN、isi_frames、dot_each 构造 dot_mask —— #
        dot_mask = torch.zeros(seq_len, dtype=torch.float32)
        pointer = 0
        for i in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            dot_mask[pointer:endp] = 1.0
            pointer = endp
            if i < dotN - 1:
                pointer += isi_frames
                if pointer >= seq_len:
                    pointer = seq_len - 1

        # —— 3.4.6 再把前面得到的短序列 (长度 ts) 的 rpf, dpf 等，填充到 seq_len —— #
        # 注意：我们只在 t < ts 时才用到“rect_within_parafovea, dot_within_parafovea” 这些值，
        #      在 t >= ts 时，全部保持 0 即可（表示没有刺激、没有眼球落在这个区域）。
        rect_within_parafovea = torch.zeros(seq_len, dtype=torch.float32)
        dot_within_parafovea  = torch.zeros(seq_len, dtype=torch.float32)
        rect_within_parafovea[:ts] = rpf
        dot_within_parafovea[:ts]  = dpf

        # —— 3.4.7 对齐到最终的五列：—— #
        # 列 0：ts_mask_tm * rect_within_parafovea
        col0 = ts_mask_tm * rect_within_parafovea
        # 列 1：dot_mask * dot_within_parafovea
        col1 = dot_mask * dot_within_parafovea
        # 列 2：dotN / 11 （常数向量）
        col2 = torch.full((seq_len,), dotN / 11.0, dtype=torch.float32)
        # 列 3：set_tp （长度 seq_len），t >= set_idx 时为 1
        col3 = torch.zeros(seq_len, dtype=torch.float32)
        if set_idx < seq_len:
            col3[set_idx:] = 1.0
        # 列 4：dot_v = dotN / ts （常数向量）
        col4 = torch.full((seq_len,), dotN / ts, dtype=torch.float32)

        # 最后把它们沿 dim=1 堆叠成 (seq_len, 5)
        u_new = torch.stack([col0, col1, col2, col3, col4], dim=1)

        # —— 3.4.8 构造 label —— #
        key = (round(duration_1, 2), dotN)
        label = label_mapping.get(key, -1)
        if label == -1:
            print(f"⚠️ 无法为 Trial_index={trial_idx} 找到 label，key={key}，设置为 -1。")

        # —— 3.4.9 把这一 trial 的结果 append 进 new_dataset —— #
        new_dataset.append({
            "u":             u_new,
            "ts":            ts,
            "set_idx":       set_idx,
            "set_idx_noisy": tm_noisy,
            "seq_len":       seq_len,
            "label":       label,      # 如果你要用 label，就打开这一行并保证 label_mapping 无误
            "dotN":          dotN,
            "Trial_index":   trial_idx,
        })

# —— 4）全部处理完毕后，将 new_dataset 保存到磁盘 —— #
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(new_dataset, output_file)
print(f"✅ 已生成并保存新数据，共 {len(new_dataset)} 条，路径：{output_file}")



# # %%
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# # 请根据实际路径修改
# data_path = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_dotN.pt"

# # 参数设置
# A = 3.0
# alpha = 2.8

# # 加载数据集
# dataset = torch.load(data_path)

# # 用于存储所有 trial 的 f(t) 值
# all_f_values = []

# # 遍历所有 trial
# for sample in dataset:
#     ts = int(sample["ts"])
#     # t 从 0 到 ts，共 ts+1 个点
#     t_array = np.arange(0, ts + 1, dtype=np.float64)
#     f_array = A * (np.exp(t_array / alpha) - 1.0)
#     all_f_values.append(f_array)

# # 合并成一个一维数组
# all_f_values = np.concatenate(all_f_values)

# # 计算分位数以便观察分布
# percentiles = np.percentile(all_f_values, [0, 25, 50, 75, 100])

# # 打印分位数
# print("f(t) 值分布的统计量：")
# print(f"最小值 (0%): {percentiles[0]:.6f}")
# print(f"第 25 百分位: {percentiles[1]:.6f}")
# print(f"中位数 (50%): {percentiles[2]:.6f}")
# print(f"第 75 百分位: {percentiles[3]:.6f}")
# print(f"最大值 (100%): {percentiles[4]:.6f}")

# # 绘制直方图
# plt.figure(figsize=(8, 4))
# plt.hist(all_f_values, bins=100)
# plt.title("所有 trial 上 f(t) 值的直方图")
# plt.xlabel("f(t)")
# plt.ylabel("频数")
# plt.tight_layout()
# plt.show()

# %%
# %%
import os
import pandas as pd
import numpy as np
import torch

# -------------------- 配置区，请根据实际改 --------------------
pre_folder      = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
resample_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/resample"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_EM_3c_settp.pt"

weber_fraction = 0.15    # 你原来用的 Weber fraction
dot_each       = 9       # 每个 dot 持续帧数（固定不变）
# --------------------------------------------------------------

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

# 1）先定义一个“小工具”函数：把 '0,1,0,…' 或 '[0,1,0,…]' 这种字符串解析成长度为 length 的列表
def parse_series(val, length, dtype=float):
    """
    - 如果 val 是形如 "[0,1,0,1]" 或 "0,1,0,1" 的 str，就拆分、转换成 list[dtype] 并返回。
    - 如果 val 是一个标量（np.isscalar）或无法解析，就直接返回 [dtype(val)]*length 或 [0]*length。
    """
    if isinstance(val, str):
        try:
            s = val.strip('[]').split(',')
            return [dtype(x) for x in s]
        except:
            return [dtype(0)] * length
    elif np.isscalar(val):
        return [dtype(val)] * length
    else:
        return [dtype(0)] * length

# 2）构建一个空列表，用来一次性把所有 trial 的最终结果都 append 进来
new_dataset = []

# 3）遍历 pre_folder 下的所有 CSV 文件
for fname in os.listdir(pre_folder):
    if not fname.endswith(".csv"):
        continue

    path_pre = os.path.join(pre_folder, fname)
    path_res = os.path.join(resample_folder, fname)
    if not os.path.exists(path_res):
        print(f"⚠️ 找不到对应的 resample 文件：{path_res}，跳过此文件。")
        continue

    # —— 3.1 读取 pre 和 resample 两张表格 —— #
    df_pre = pd.read_csv(path_pre)
    df_res = pd.read_csv(path_res)

    # —— 3.2 在 pre 表格中，根据 'block' 累计计算全局 Trial_index —— #
    block_counts = df_pre.groupby('block').size().sort_index()
    offset_map = {}   # 存 block → 当前 block 的起始 trial_idx 偏移量
    cum = 0
    for blk in block_counts.index:
        offset_map[blk] = cum
        cum += block_counts[blk]

    df_pre['Trial_index'] = df_pre.apply(
        lambda r: offset_map[r['block']] + int(r['trial']), axis=1
    )

    # —— 3.3 在 resample 表格中，也生成同样的全局 Trial_index 列 —— #
    if 'block' in df_res.columns and 'trial' in df_res.columns:
        rc = df_res.groupby('block').size().sort_index()
        off_r = {}
        c = 0
        for b in rc.index:
            off_r[b] = c
            c += rc[b]
        df_res['Trial_index'] = df_res.apply(
            lambda r: off_r[r['block']] + int(r['trial']), axis=1
        )
    else:
        # 如果 resample 原本就有 trial_index，就重命名为统一字段
        if 'trial_index' in df_res.columns:
            df_res.rename(columns={'trial_index':'Trial_index'}, inplace=True)
        else:
            # 万一两个都没有，那就只能按行索引对应了（不推荐）
            print(f"❌ Warning: {fname} 的 resample 表里没有 block/trial 也没有 trial_index！")

    # —— 3.4 遍历 pre 表格里的每一行 trial，构建新的 u_new —— #
    for _, row in df_pre.iterrows():
        ts         = int(row['ts'])            # 刺激时长（帧）
        seq_len    = int(row['seq_len'])       # 序列总长度
        dotN       = int(row['dotN_1'])        # dot 总数（6 or 11）
        set_idx    = int(row['set_idx'])       # reproduction 开始的索引
        trial_idx  = int(row['Trial_index'])
        duration_1 = float(row['duration_1'])  # 用来映射 label 的 key
        isi_frames = int(row['isi_1'])         # 该 trial 的 dot 间隔帧数（动态读取）

        # —— 3.4.1 生成 noisy_tm —— #
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        # 限制 tm 在 [1, seq_len-1] 之间的整数
        tm_noisy = max(1, min(seq_len - 1, int(round(tm))))

        # —— 3.4.2 找到该 trial 在 resample 表里的对应行 —— #
        sub = df_res[df_res['Trial_index'] == trial_idx]
        if sub.empty:
            print(f"⚠️ 在 resample 中没找到 Trial_index = {trial_idx}（文件 {fname}），跳过此 trial。")
            continue
        cres = sub.iloc[0]

        # —— 3.4.3 解析 resample 表里的几列眼动序列 —— #
        # parse_series 会返回一个长度为 ts 的 Python list，我们再转成 Tensor
        display_box = torch.tensor(parse_series(cres['display_box'], ts, dtype=int), dtype=torch.float32)
        sac          = torch.tensor(parse_series(cres['has_saccade'], ts), dtype=torch.float32)
        fix          = torch.tensor(parse_series(cres['has_fixation'], ts), dtype=torch.float32)
        blk          = torch.tensor(parse_series(cres['has_blink'], ts), dtype=torch.float32)
        rpf          = torch.tensor(parse_series(cres['rect_within_parafovea'], ts), dtype=torch.float32)
        dpf          = torch.tensor(parse_series(cres['dot_within_parafovea'], ts), dtype=torch.float32)

        # —— 3.4.4 先根据 tm_noisy 构造 ts_mask_tm —— #
        # ts_mask_tm: 对于 t < tm_noisy, 取 1；否则 0；长度是 seq_len
        ts_mask_tm = torch.zeros(seq_len, dtype=torch.float32)
        ts_cut     = min(tm_noisy, seq_len)  # 防止 tm_noisy 跑出范围
        ts_mask_tm[:ts_cut] = 1.0

        # —— 3.4.5 根据 dotN、isi_frames、dot_each 构造 dot_mask —— #
        dot_mask = torch.zeros(seq_len, dtype=torch.float32)
        pointer = 0
        for i in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            dot_mask[pointer:endp] = 1.0
            pointer = endp
            if i < dotN - 1:
                pointer += isi_frames
                if pointer >= seq_len:
                    pointer = seq_len - 1

        # —— 3.4.6 再把前面得到的短序列 (长度 ts) 的 rpf, dpf 等，填充到 seq_len —— #
        # 注意：我们只在 t < ts 时才用到“rect_within_parafovea, dot_within_parafovea” 这些值，
        #      在 t >= ts 时，全部保持 0 即可（表示没有刺激、没有眼球落在这个区域）。
        rect_within_parafovea = torch.zeros(seq_len, dtype=torch.float32)
        dot_within_parafovea  = torch.zeros(seq_len, dtype=torch.float32)
        rect_within_parafovea[:ts] = rpf
        dot_within_parafovea[:ts]  = dpf

        # —— 3.4.7 对齐到最终的五列：—— #
        # 列 0：ts_mask_tm * rect_within_parafovea
        col0 = ts_mask_tm * rect_within_parafovea
        # 列 1：dot_mask * dot_within_parafovea
        col1 = dot_mask * dot_within_parafovea
        # 列 2：dotN / 11 （常数向量）
        col2 = torch.full((seq_len,), dotN / 11.0, dtype=torch.float32)
        # 列 3：set_tp （长度 seq_len），t >= set_idx 时为 1
        col3 = torch.zeros(seq_len, dtype=torch.float32)
        if set_idx < seq_len:
            col3[set_idx:] = 1.0
        # 列 4：dot_v = dotN / ts （常数向量）
        col4 = torch.full((seq_len,), dotN / ts, dtype=torch.float32)

        # 最后把它们沿 dim=1 堆叠成 (seq_len, 5)
        # u_new = torch.stack([col0, col1, col2, col3, col4], dim=1)
        # 只要前两列
        u_new = torch.stack([col0, col1, col3], dim=1)

        # —— 3.4.8 构造 label —— #
        key = (round(duration_1, 2), dotN)
        label = label_mapping.get(key, -1)
        if label == -1:
            print(f"⚠️ 无法为 Trial_index={trial_idx} 找到 label，key={key}，设置为 -1。")

        # —— 3.4.9 把这一 trial 的结果 append 进 new_dataset —— #
        new_dataset.append({
            "u":             u_new,
            "ts":            ts,
            "set_idx":       set_idx,
            "set_idx_noisy": tm_noisy,
            "seq_len":       seq_len,
            "label":       label,      # 如果你要用 label，就打开这一行并保证 label_mapping 无误
            "dotN":          dotN,
            "Trial_index":   trial_idx,
        })

# —— 4）全部处理完毕后，将 new_dataset 保存到磁盘 —— #
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(new_dataset, output_file)
print(f"✅ 已生成并保存新数据，共 {len(new_dataset)} 条，路径：{output_file}")

# %%
# %%
import os
import pandas as pd
import numpy as np
import torch

# -------------------- 配置区，请根据实际改 --------------------
pre_folder      = "/Users/juntao/Desktop/proj_TimePerception/original_data/iti_frames"
resample_folder = "/Users/juntao/Desktop/proj_TimePerception/original_data/resample"
output_file     = "/Users/juntao/Desktop/proj_TimePerception/data/dataset_EM_4c_settp_dotv.pt"

weber_fraction = 0.15    # 你原来用的 Weber fraction
dot_each       = 9       # 每个 dot 持续帧数（固定不变）
# --------------------------------------------------------------

# label 映射
label_mapping = {
    (2.48, 6): 0, (2.48, 11): 1,
    (3.98, 6): 2, (3.98, 11): 3,
    (4.48, 6): 4, (4.48, 11): 5,
    (4.98, 6): 6, (4.98, 11): 7,
}

# 1）先定义一个“小工具”函数：把 '0,1,0,…' 或 '[0,1,0,…]' 这种字符串解析成长度为 length 的列表
def parse_series(val, length, dtype=float):
    """
    - 如果 val 是形如 "[0,1,0,1]" 或 "0,1,0,1" 的 str，就拆分、转换成 list[dtype] 并返回。
    - 如果 val 是一个标量（np.isscalar）或无法解析，就直接返回 [dtype(val)]*length 或 [0]*length。
    """
    if isinstance(val, str):
        try:
            s = val.strip('[]').split(',')
            return [dtype(x) for x in s]
        except:
            return [dtype(0)] * length
    elif np.isscalar(val):
        return [dtype(val)] * length
    else:
        return [dtype(0)] * length

# 2）构建一个空列表，用来一次性把所有 trial 的最终结果都 append 进来
new_dataset = []

# 3）遍历 pre_folder 下的所有 CSV 文件
for fname in os.listdir(pre_folder):
    if not fname.endswith(".csv"):
        continue

    path_pre = os.path.join(pre_folder, fname)
    path_res = os.path.join(resample_folder, fname)
    if not os.path.exists(path_res):
        print(f"⚠️ 找不到对应的 resample 文件：{path_res}，跳过此文件。")
        continue

    # —— 3.1 读取 pre 和 resample 两张表格 —— #
    df_pre = pd.read_csv(path_pre)
    df_res = pd.read_csv(path_res)

    # —— 3.2 在 pre 表格中，根据 'block' 累计计算全局 Trial_index —— #
    block_counts = df_pre.groupby('block').size().sort_index()
    offset_map = {}   # 存 block → 当前 block 的起始 trial_idx 偏移量
    cum = 0
    for blk in block_counts.index:
        offset_map[blk] = cum
        cum += block_counts[blk]

    df_pre['Trial_index'] = df_pre.apply(
        lambda r: offset_map[r['block']] + int(r['trial']), axis=1
    )

    # —— 3.3 在 resample 表格中，也生成同样的全局 Trial_index 列 —— #
    if 'block' in df_res.columns and 'trial' in df_res.columns:
        rc = df_res.groupby('block').size().sort_index()
        off_r = {}
        c = 0
        for b in rc.index:
            off_r[b] = c
            c += rc[b]
        df_res['Trial_index'] = df_res.apply(
            lambda r: off_r[r['block']] + int(r['trial']), axis=1
        )
    else:
        # 如果 resample 原本就有 trial_index，就重命名为统一字段
        if 'trial_index' in df_res.columns:
            df_res.rename(columns={'trial_index':'Trial_index'}, inplace=True)
        else:
            # 万一两个都没有，那就只能按行索引对应了（不推荐）
            print(f"❌ Warning: {fname} 的 resample 表里没有 block/trial 也没有 trial_index！")

    # —— 3.4 遍历 pre 表格里的每一行 trial，构建新的 u_new —— #
    for _, row in df_pre.iterrows():
        ts         = int(row['ts'])            # 刺激时长（帧）
        seq_len    = int(row['seq_len'])       # 序列总长度
        dotN       = int(row['dotN_1'])        # dot 总数（6 or 11）
        set_idx    = int(row['set_idx'])       # reproduction 开始的索引
        trial_idx  = int(row['Trial_index'])
        duration_1 = float(row['duration_1'])  # 用来映射 label 的 key
        isi_frames = int(row['isi_1'])         # 该 trial 的 dot 间隔帧数（动态读取）

        # —— 3.4.1 生成 noisy_tm —— #
        tm = np.random.normal(loc=ts, scale=weber_fraction * ts)
        # 限制 tm 在 [1, seq_len-1] 之间的整数
        tm_noisy = max(1, min(seq_len - 1, int(round(tm))))

        # —— 3.4.2 找到该 trial 在 resample 表里的对应行 —— #
        sub = df_res[df_res['Trial_index'] == trial_idx]
        if sub.empty:
            print(f"⚠️ 在 resample 中没找到 Trial_index = {trial_idx}（文件 {fname}），跳过此 trial。")
            continue
        cres = sub.iloc[0]

        # —— 3.4.3 解析 resample 表里的几列眼动序列 —— #
        # parse_series 会返回一个长度为 ts 的 Python list，我们再转成 Tensor
        display_box = torch.tensor(parse_series(cres['display_box'], ts, dtype=int), dtype=torch.float32)
        sac          = torch.tensor(parse_series(cres['has_saccade'], ts), dtype=torch.float32)
        fix          = torch.tensor(parse_series(cres['has_fixation'], ts), dtype=torch.float32)
        blk          = torch.tensor(parse_series(cres['has_blink'], ts), dtype=torch.float32)
        rpf          = torch.tensor(parse_series(cres['rect_within_parafovea'], ts), dtype=torch.float32)
        dpf          = torch.tensor(parse_series(cres['dot_within_parafovea'], ts), dtype=torch.float32)

        # —— 3.4.4 先根据 tm_noisy 构造 ts_mask_tm —— #
        # ts_mask_tm: 对于 t < tm_noisy, 取 1；否则 0；长度是 seq_len
        ts_mask_tm = torch.zeros(seq_len, dtype=torch.float32)
        ts_cut     = min(tm_noisy, seq_len)  # 防止 tm_noisy 跑出范围
        ts_mask_tm[:ts_cut] = 1.0

        # —— 3.4.5 根据 dotN、isi_frames、dot_each 构造 dot_mask —— #
        dot_mask = torch.zeros(seq_len, dtype=torch.float32)
        pointer = 0
        for i in range(dotN):
            endp = min(pointer + dot_each, seq_len)
            dot_mask[pointer:endp] = 1.0
            pointer = endp
            if i < dotN - 1:
                pointer += isi_frames
                if pointer >= seq_len:
                    pointer = seq_len - 1

        # —— 3.4.6 再把前面得到的短序列 (长度 ts) 的 rpf, dpf 等，填充到 seq_len —— #
        # 注意：我们只在 t < ts 时才用到“rect_within_parafovea, dot_within_parafovea” 这些值，
        #      在 t >= ts 时，全部保持 0 即可（表示没有刺激、没有眼球落在这个区域）。
        rect_within_parafovea = torch.zeros(seq_len, dtype=torch.float32)
        dot_within_parafovea  = torch.zeros(seq_len, dtype=torch.float32)
        rect_within_parafovea[:ts] = rpf
        dot_within_parafovea[:ts]  = dpf

        # —— 3.4.7 对齐到最终的五列：—— #
        # 列 0：ts_mask_tm * rect_within_parafovea
        col0 = ts_mask_tm * rect_within_parafovea
        # 列 1：dot_mask * dot_within_parafovea
        col1 = dot_mask * dot_within_parafovea
        # 列 2：dotN / 11 （常数向量）
        col2 = torch.full((seq_len,), dotN / 11.0, dtype=torch.float32)
        # 列 3：set_tp （长度 seq_len），t >= set_idx 时为 1
        col3 = torch.zeros(seq_len, dtype=torch.float32)
        if set_idx < seq_len:
            col3[set_idx:] = 1.0
        # 列 4：dot_v = dotN / ts （常数向量）
        col4 = torch.full((seq_len,), dotN / ts, dtype=torch.float32)

        # 最后把它们沿 dim=1 堆叠成 (seq_len, 5)
        # u_new = torch.stack([col0, col1, col2, col3, col4], dim=1)
        # 只要前两列
        u_new = torch.stack([col0, col1, col3, col4], dim=1)

        # —— 3.4.8 构造 label —— #
        key = (round(duration_1, 2), dotN)
        label = label_mapping.get(key, -1)
        if label == -1:
            print(f"⚠️ 无法为 Trial_index={trial_idx} 找到 label，key={key}，设置为 -1。")

        # —— 3.4.9 把这一 trial 的结果 append 进 new_dataset —— #
        new_dataset.append({
            "u":             u_new,
            "ts":            ts,
            "set_idx":       set_idx,
            "set_idx_noisy": tm_noisy,
            "seq_len":       seq_len,
            "label":       label,      # 如果你要用 label，就打开这一行并保证 label_mapping 无误
            "dotN":          dotN,
            "Trial_index":   trial_idx,
        })

# —— 4）全部处理完毕后，将 new_dataset 保存到磁盘 —— #
os.makedirs(os.path.dirname(output_file), exist_ok=True)
torch.save(new_dataset, output_file)
print(f"✅ 已生成并保存新数据，共 {len(new_dataset)} 条，路径：{output_file}")
# %%
