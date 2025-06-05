import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from model import FiringRateRNN

def inference(model: FiringRateRNN,
                  trials_list: list[dict],
                  device: str = 'cpu',
                  add_noise: bool = False,
                  threshold_mode: str = 'fixed',
                  threshold_value: float = 1.0):
    """
    一次 pass 完成：
      1) per-trial inference 结果保存在 results_list
      2) 按 label 汇总 z_t 并计算对齐后的 mean/std 保存在 z_summary

    参数:
      threshold_mode: 'fixed' 或 'max'
        - 'fixed': 直接用 threshold_value
        - 'ramp' : 用 generate_ramp_target 计算出的 ramp target 峰值
      threshold_value: 当 threshold_mode='fixed' 时使用的阈值

    返回:
      {
        'inference': results_list,
        'z_timecourses': z_summary
      }
    """
    model.to(device).eval()

    results_list = []
    z_by_label   = defaultdict(list)
    post_lens    = defaultdict(list)

    with torch.no_grad():
        for trial in tqdm(trials_list, desc="Inference", unit="trial"):
            # 读取 trial
            u      = trial["u"].unsqueeze(0).to(device)   # (1,T,input_dim)
            set_idx= trial["set_idx"]
            ts     = trial["ts"]
            label  = trial["label"]
            dotN   = trial["dotN"]
            T      = u.shape[1]

            x = torch.zeros(1, model.hidden_size, device=device)

            firing_rates = []
            z_seq        = []

            for t in range(T):
                r   = torch.tanh(x)
                z_t = (r @ model.W_out).squeeze(-1) + model.b_out

                firing_rates.append(r.squeeze(0).cpu().numpy())
                z_seq.append(z_t.item())

                if add_noise:
                    noise = torch.randn_like(x) * model.noise_std
                else:
                    noise = torch.zeros_like(x)

                dx = (
                    -x
                    + r @ model.W_rec.T
                    + u[:,t,:] @ model.W_in.T
                    + model.b_rec
                    + noise
                ) * (model.dt / model.tau)
                x = x + dx

            firing_rates = np.stack(firing_rates, axis=0)  # (T,hidden)
            z_seq         = np.array(z_seq)               # (T,)

            # 计算阈值
            if threshold_mode == 'max':
                from inference import generate_ramp_target
                targ = generate_ramp_target([ts],[set_idx],[T],[dotN], device=device)
                targ = targ.squeeze(0).cpu().numpy()
                threshold = np.max(targ)
            else:
                threshold = threshold_value

            # 取 production phase 并找 Tp
            post = z_seq[set_idx+1:]
            idx  = np.where(post >= threshold)[0]
            Tp   = int(idx[0]) if idx.size else None

            results_list.append({
                "firing_rate": firing_rates,
                "z_seq":       z_seq,
                "Tp":          Tp,
                "ts":          ts,
                "label":       label,
                "set_idx":     set_idx,
                "seq_len":     T
            })

            z_by_label[label].append(post)
            post_lens[label].append(len(post))

    # z_summary 汇总
    z_summary = {}
    for label, seqs in z_by_label.items():
        max_len = max(post_lens[label])
        aligned = np.full((len(seqs), max_len), np.nan)
        for i, arr in enumerate(seqs):
            aligned[i, :len(arr)] = arr
        mean = np.nanmean(aligned, axis=0)
        std  = np.nanstd(aligned, axis=0)
        time = np.arange(1, max_len+1)
        z_summary[label] = {
            "z_seqs": seqs,
            "mean":   mean,
            "std":    std,
            "time":   time
        }

    return {
        "inference":     results_list,
        "z_timecourses": z_summary
    }