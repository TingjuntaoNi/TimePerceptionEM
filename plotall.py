#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_firing_rates(results, duration_list, dotN_list, duration_to_color, units, dot_each=9):
    """绘制随机选中神经元的 firing rate 曲线（Set 对齐）"""
    for unit in units:
        # 每个 label 的 firing rate 列表
        label_firing = {l: [] for l in range(8)}
        for trial in results:
            firing = trial["firing_rate"]  # (T, hidden_size)
            label  = trial["label"]
            set_idx= trial["set_idx"]
            label_firing[label].append(firing[set_idx+1:, unit])

        plt.figure(figsize=(8,5))
        for l in range(8):
            arrs = label_firing[l]
            if not arrs:
                continue
            max_len = max(len(a) for a in arrs)
            aligned = np.full((len(arrs), max_len), np.nan)
            for i,a in enumerate(arrs):
                aligned[i,:len(a)] = a
            mean = np.nanmean(aligned, axis=0)
            std  = np.nanstd(aligned, axis=0)
            x = np.arange(len(mean))

            dur  = duration_list[l]
            dotN = dotN_list[l]
            base_color = duration_to_color[dur]
            if dotN == 6:
                color, ls, alpha = base_color, '--', 0.25
            else:
                # 浅色
                rgb = np.array(mcolors.to_rgb(base_color))
                white = np.ones(3)
                light = tuple(rgb + (white-rgb)*0.6)
                color, ls, alpha = light, '-', 0.2

            plt.plot(x, mean, color=color, linestyle=ls,
                     label=f'{dur}s, dotN={dotN}')
            plt.fill_between(x, mean-std, mean+std,
                             color=color, alpha=alpha)

        plt.title(f'Unit {unit} firing rates (Set‑aligned)')
        plt.xlabel('Time since Set (frames)')
        plt.ylabel('Firing rate (tanh(x))')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', frameon=False)
        plt.tight_layout()
        plt.show()


def plot_tp_scatter(results, duration_list, dotN_list, duration_to_color):
    """绘制 Tp vs ts 的散点图"""
    # 收集 Tp 和 ts
    label_Tp = {l: [] for l in range(8)}
    label_ts = {l: [] for l in range(8)}
    for e in results:
        Tp, ts, l = e["Tp"], e["ts"], e["label"]
        if Tp is None: continue
        label_Tp[l].append(Tp)
        label_ts[l].append(ts)
    mean_Tp = [np.nan if not label_Tp[l] else np.mean(label_Tp[l]) for l in range(8)]
    mean_ts = [np.nan if not label_ts[l] else np.mean(label_ts[l]) for l in range(8)]

    fig, ax = plt.subplots(figsize=(5,5))
    for l in range(8):
        x,y = mean_ts[l], mean_Tp[l]
        if np.isnan(x) or np.isnan(y): continue
        dur, dotN = duration_list[l], dotN_list[l]
        color = duration_to_color[dur]
        alpha = 0.6 if dotN==6 else 0.3
        edgekw = {'edgecolors':'none'}
        ax.scatter(x, y, color=color, alpha=alpha,
                   s=80, **edgekw, label=f'{dur}s, dotN={dotN}')

    valid_x = [x for x in mean_ts if not np.isnan(x)]
    valid_y = [y for y in mean_Tp if not np.isnan(y)]
    mn, mx = min(valid_x+valid_y), max(valid_x+valid_y)
    ax.plot([mn,mx],[mn,mx],'k--')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Trained interval (frames)")
    ax.set_ylabel("Tp (frames)")
    ax.set_title("Tp vs Trained interval")
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', frameon=False)
    plt.tight_layout()
    plt.show()


def plot_z_timecourses(z_summary, duration_list, dotN_list, duration_to_color):
    """绘制每个条件下 z_t 随时间的变化"""
    labels_sorted = sorted(range(8), key=lambda l:(duration_list[l], dotN_list[l]))
    plt.figure(figsize=(8,5))
    for l in labels_sorted:
        entry = z_summary[l]
        t, mean, std = entry['time'], entry['mean'], entry['std']
        dur, dotN = duration_list[l], dotN_list[l]
        color = duration_to_color[dur]
        if dotN==6:
            ls, alpha = '--', 0.5
        else:
            ls, alpha = '-', 0.2
        plt.plot(t, mean, color=color, linestyle=ls,
                 label=f'{dur}s, dotN={dotN}')
        plt.fill_between(t, mean-std, mean+std,
                         color=color, alpha=alpha)
    plt.xlabel("Time since Set (frames)")
    plt.ylabel("Network output $z_t$")
    plt.title("$z$ over time for each condition")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small', frameon=False)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot all figures from a saved inference .pkl")
    parser.add_argument('infile', type=str, help="path to your .pkl file (must contain keys 'inference' and 'z_timecourses')")
    args = parser.parse_args()

    # load
    data = pickle.load(open(args.infile,'rb'))
    results   = data['inference']
    z_summary = data['z_timecourses']

    # define mapping
    duration_to_color = {2.48:'purple', 3.98:'navy', 4.48:'green', 4.98:'gold'}
    duration_list = [2.48,2.48,3.98,3.98,4.48,4.48,4.98,4.98]
    dotN_list     = [   6,   11,    6,    11,    6,    11,    6,    11]

    # choose two random units
    np.random.seed(42)
    num_units = len(results[0]['firing_rate'][0])  # hidden_size
    units = np.random.choice(num_units, size=2, replace=False)
    print("Selected units:", units)

    # Plot all
    plot_firing_rates(results, duration_list, dotN_list, duration_to_color, units)
    plot_tp_scatter  (results, duration_list, dotN_list, duration_to_color)
    plot_z_timecourses(z_summary, duration_list, dotN_list, duration_to_color)

if __name__ == '__main__':
    import argparse
    main()
