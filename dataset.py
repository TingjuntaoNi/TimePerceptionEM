import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ramp import generate_ramp_target
from sklearn.model_selection import StratifiedKFold
import json
from torch.utils.data import Subset, DataLoader
import torch
from torch.utils.data import Dataset

# example
#     "u": Tensor(T_i, 2),
#     "ts": 9.0,
#     "set_idx": 5,
#     "seq_len": 25  # or len(u)
# }


class TimingDataset(torch.utils.data.Dataset):
    def __init__(self, trials_list, device):
        """
        trials_list: list of dicts with keys:
            - "u": Tensor of shape (T_i, 2)
            - "ts": float, stimulus duration (num_frames)
            - "set_idx": int, the index of the set onset (the onset time point of green box)
            - "dot_N": int, the number of dots in this trial
        """
        self.trials = trials_list
        self.device = device

    def __len__(self): # return the number of trials
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        u = trial["u"].to(self.device)  # (T_i, 2)
        ts = trial["ts"]
        set_idx = trial["set_idx"]
        seq_len = u.shape[0]
        dotN = trial["dotN"]  # new
        cond =trial["label"]  # new
        
        # # ✅ 打印调试信息，只打印前 10 个
        # if idx < 20:
        #     label_guess = trial.get("label", "N/A")  # 如果 trial 中有 label 就打印
        #     density = dotN / ts
        #     print(f"[Trial {idx}] label={label_guess}, ts={ts}, dotN={dotN}, density={density:.4f}, set_idx={set_idx}, seq_len={seq_len}")

        target = generate_ramp_target(
            ts_list=[ts],
            set_idx_list=[set_idx],
            seq_len_list=[seq_len],
            dotN_list=[dotN],  # new
            device=self.device
        )[0]

        return u, target, set_idx, dotN, cond


def collate_variable_length(batch):
    """
    batch: list of tuples (u_i, target_i), each with shape (T_i, 2) and (T_i,)
    Returns:
        - padded_u: (batch, T_max, 2)
        - padded_target: (batch, T_max)
        - lengths: list of original T_i
    """

    us, targets, set_idxs, dotNs, labels = zip(*batch)
    lengths = [u.shape[0] for u in us] # original lengths

    padded_u = pad_sequence(us, batch_first=True)         # shape: (batch, T_max, 2)
    padded_target = pad_sequence(targets, batch_first=True)  # shape: (batch, T_max)
    set_idx_batch = torch.tensor(set_idxs)  # set_idx batch, shape = (batch_size,)
    dotN_batch = torch.tensor(dotNs) # dotN batch, shape = (batch_size,)
    label_batch = torch.tensor(labels)  

    return padded_u, padded_target, set_idx_batch, dotN_batch, torch.tensor(lengths), label_batch





# class TimingDataset(Dataset):
#     """
#     PyTorch Dataset for timing RNN trials.

#     Each trial dict should contain:
#       - 'u':        torch.Tensor of shape (T, input_dim)
#       - 'ts_true':  true stimulus duration (in frames)
#       - 'set_idx_noisy': noisy production onset index
#       - 'seq_len':  total trial length (in frames)
#       - 'dotN':     number of dots
#     """
#     def __init__(self, trials_list, device='cpu', A=3.0, alpha=2.8):
#         self.trials = trials_list
#         self.device = device
#         self.A = A
#         self.alpha = alpha

#     def __len__(self):
#         return len(self.trials)

#     def __getitem__(self, idx):
#         trial = self.trials[idx]
#         # input sequence (T, input_dim)
#         u = trial['u'].to(self.device)
#         # true stimulus duration
#         ts = trial.get('ts_true', trial.get('ts'))
#         # noisy production onset
#         set_idx = trial.get('set_idx_noisy', trial.get('set_idx'))
#         # number of dots
#         dotN = trial['dotN']
#         # sequence length
#         seq_len = trial['seq_len']

#         # generate ramp target: starts at set_idx+1, shape determined by ts
#         target = generate_ramp_target(
#             ts_list=[ts],
#             set_idx_noisy_list=[set_idx],
#             seq_len_list=[seq_len],
#             dotN_list=[dotN],
#             A=self.A,
#             alpha=self.alpha,
#             device=self.device
#         )[0]

#         # return u, target ramp, noisy onset, dot count, original length
#         return u, target, set_idx, dotN, seq_len


# def collate_variable_length(batch):
#     """
#     Collate function to pad variable-length sequences in a batch.

#     batch: list of tuples (u, target, set_idx, dotN, seq_len)
#     Returns:
#       - padded_u:      (B, T_max, input_dim)
#       - padded_target: (B, T_max)
#       - set_idx_batch: (B,)
#       - dotN_batch:    (B,)
#       - lengths:       (B,)
#     """
#     us, targets, set_idxs, dotNs, lengths = zip(*batch)
#     T_max = max(lengths)

#     # pad u and target
#     padded_u = pad_sequence(us, batch_first=True)
#     padded_target = pad_sequence(targets, batch_first=True)

#     set_idx_batch = torch.tensor(set_idxs, dtype=torch.long)
#     dotN_batch    = torch.tensor(dotNs,    dtype=torch.long)
#     length_batch  = torch.tensor(lengths,  dtype=torch.long)

#     return padded_u, padded_target, set_idx_batch, dotN_batch, length_batch







def build_index_with_kfold(trials_list, n_splits=5, random_state=42, save_path=None):
    """
    Generate stratified k-fold indices (based on trial labels) and optionally save them.

    Parameters:
        trials_list: List of trial dictionaries.
        n_splits: Number of folds for cross-validation.
        random_state: Random seed for reproducibility.
        save_path: If provided, saves the index sets as a JSON file.

    Returns:
        index_sets: A list containing the train and validation indices for each fold.
    """

    # Extract labels to ensure stratified splitting across different experimental conditions
    labels = [trial["label"] for trial in trials_list]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    index_sets = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(trials_list,labels)):
        index_sets.append({
            "fold": fold,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist()
        })

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(index_sets, f, indent=2)

    return index_sets


def get_fold_dataloaders(dataset, index_sets, fold, batch_size, collate_fn):
    """
    Create training and validation DataLoaders for the given fold.

    Parameters:
        dataset: The original TimingDataset object.
        index_sets: List of index dictionaries generated by build_index_with_kfold().
        fold: The current fold to use (0-based index).
        batch_size: Batch size for the DataLoader.
        collate_fn: Collate function to handle variable-length sequences.

    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
    """
    train_idx = index_sets[fold]["train_idx"]
    val_idx = index_sets[fold]["val_idx"]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader
