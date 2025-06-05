import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def generate_ramp_target(
    ts_list: list[float],
    set_idx_list: list[int],
    seq_len_list: list[int],
    dotN_list,
    A: float = 3, # default is 3.0
    alpha: float = 2.8,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Generate target ramp outputs for a batch of trials based on:
        f_i(t) = A * (exp(t_rel / (alpha * t_s)) - 1),
    defined only after the production onset.

    Parameters:
        ts_list (list of float): Stimulus durations (in frames) for each trial.
        set_idx_list (list of int): The index of the Set onset (i.e., end of stimulus) for each trial.
        seq_len_list (list of int): Total number of time steps (trial length) for each trial.
        A (float): ramp amplitude.
        alpha (float): scaling factor controlling the ramp curvature.
        device (torch.device): device for the output tensor.

    Returns:
        targets(torch.Tensor): shape (batch, seq_len) target outputs,
                      zeros before Set, ramp thereafter.
    """
    ramp_list = []

    for ts, set_idx, seq_len, dotN in zip(ts_list, set_idx_list, seq_len_list,dotN_list):
        target = torch.zeros(seq_len, device=device)

        t_rel = torch.arange(1, seq_len - set_idx, device=device).float()  # (T_rel,)
        exponent = t_rel / (alpha * ts)
        ramp = A * (exponent.exp() - 1)

        target[set_idx + 1:] = ramp
        ramp_list.append(target)

    # Pad to (batch, max_seq_len)
    return pad_sequence(ramp_list, batch_first=True)

# tm
# def generate_ramp_target(
#     ts_list: list[float],
#     set_idx_noisy_list: list[int],
#     seq_len_list: list[int],
#     dotN_list,
#     A: float = 3.0,
#     alpha: float = 2.8,
#     device: torch.device = torch.device("cpu")
# ) -> torch.Tensor:
#     """
#     Generate a batch of ramp targets:
#       - ramp shape f(t) = A * (exp(t_rel/(alpha*ts)) - 1)
#       - ramp starts at set_idx_noisy + 1
#       - uses true ts for shape

#     Returns:
#       targets: Tensor of shape (batch, max_seq_len)
#     """
#     ramps = []
#     for ts, set_idx_noisy, seq_len,dotN in zip(ts_list, set_idx_noisy_list, seq_len_list,dotN_list):
#         # initialize zeros
#         target = torch.zeros(seq_len, device=device, dtype=torch.float32)

#         # compute relative time after noisy set
#         T_post = seq_len - (set_idx_noisy + 1)
#         if T_post > 0:
#             t_rel = torch.arange(1, T_post + 1, device=device).float()
#             ramp  = A * (t_rel.div(alpha * ts).exp() - 1)
#             # place into target
#             target[set_idx_noisy + 1 : set_idx_noisy + 1 + T_post] = ramp

#         ramps.append(target)

#     # pad to (batch, max_seq_len)
#     return pad_sequence(ramps, batch_first=True)



# # 新增：dotN 对幅度的影响
# def generate_ramp_target(
#     ts_list, 
#     set_idx_list, 
#     seq_len_list, 
#     dotN_list,
#     A:float=1.5, 
#     alpha:float=2.8, 
#     beta:float=1.5,
#     device=torch.device("cpu")
# ):
#     """
#     ramp(t) = A*(1 + beta*(1/d)) * (exp(t_rel/(alpha*ts)) - 1)
#     """
#     ramp_list = []
#     for ts, set_idx, seq_len, dotN in zip(ts_list, set_idx_list, seq_len_list, dotN_list):
#         target = torch.zeros(seq_len, device=device)
#         t_rel = torch.arange(1, seq_len - set_idx, device=device).float()
#         base  = (t_rel / (alpha * ts)).exp() - 1  # 原始指数项
#         mod   = 1.0 + beta * (1.0 / dotN)         # 新增调制因子
#         ramp  = A * mod * base
#         target[set_idx + 1:] = ramp
#         ramp_list.append(target)
#     return pad_sequence(ramp_list, batch_first=True)


# 以下ramp函数模型在不同条件下Tp的复现效果不好
# # 新增：dotN 对幅度的影响
# def generate_ramp_target(
#     ts_list,
#     set_idx_list,
#     seq_len_list,
#     dotN_list,  # 新增
#     A:float = 1.5,
#     alpha: float = 2.8,
#     gamma: float = 0.1,  # dotN 对幅度的影响
#     device=torch.device("cpu")
# ):
#     ramp_list = []

#     for ts, set_idx, seq_len, dotN in zip(ts_list, set_idx_list, seq_len_list, dotN_list):

#         target = torch.zeros(seq_len, device=device)
#         t_rel = torch.arange(1, seq_len - set_idx, device=device).float()
#         base  = A * (t_rel.div(alpha * ts).exp() - 1)
#         mod   = 1.0 + gamma * dotN
#         ramp  = base.div(mod)            # 除以 (1 + gamma*d)
#         target[set_idx+1:] = ramp
#         ramp_list.append(target)

#     return pad_sequence(ramp_list, batch_first=True)