from max_ent.gridworld.gridworld import Directions
from typing import NamedTuple
import numpy as np
from pathlib import Path
import json
from numpy.lib.arraysetops import setdiff1d
from scipy.spatial import distance
import math
from collections import namedtuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

Constraints = namedtuple('Constraints', ['state', 'action', 'feature'])
colors = ['red', 'orange', 'purple', 'green']


def load_data(path):
    with Path(path).open() as f:
        r = json.load(f)[0]
    return r


def get_true_cons(true_data):
    state_cons = np.argwhere(
        np.array(true_data['state_reward']) <= -50).squeeze()
    action_cons = np.argwhere(
        np.array(true_data['action_reward']) <= -50).squeeze()
    feature_cons = np.argwhere(
        np.array(true_data['color_reward']) <= -50).squeeze()

    return Constraints(state_cons, action_cons, feature_cons)


def get_predicted_cons(data, n_set, n_len):
    learned = data['learned_constraints']
    cons = []
    for i in range(n_set):
        cons.append([])
        for j in range(n_len):
            cons[i].append([])
            l = learned[i][j]
            state_cons = [x['value'] for x in l if x['type'] == 'state']
            action_cons = [x['value'] for x in l if x['type'] == 'action']
            feature_cons = [x['value'] for x in l if x['type'] == 'feature']
            cons[i][j] = Constraints(state_cons, action_cons, feature_cons)
    return cons


def fp(true, x):
    N = len(x.state) + len(x.action) + len(x.feature)
    fs = len(setdiff1d(x.state, true.state))
    fa = len(setdiff1d(x.action, true.action))
    ff = len(setdiff1d(x.feature, true.feature))
    return (fs + fa + ff) / N


def dist(x):
    seq = [list(zip(s, a)) for s, a in zip(x['state_seq'], x['action_seq'])]
    dist = np.ones((81, 8)) * 1e-6
    for t in seq:
        for s, a in t:
            dist[s, a] += 1
    return dist/dist.sum().reshape(-1, 1)


def kl(true, x):
    true_dist = dist(true)
    x_dist = dist(x)
    kl = true_dist * np.log(true_dist/x_dist)
    return kl.sum()


def get_stats(true_cons, pred_cons, true_demo, pred_demo, n_set, n_len):
    fp_list = np.zeros((n_set, n_len))
    kl_list = np.zeros((n_set, n_len))
    for i in range(n_set):
        for j in range(n_len):
            fp_list[i, j] = fp(true_cons, pred_cons[i][j])
            kl_list[i, j] = kl(true_demo[i], pred_demo[i][j][0])

    return fp_list, kl_list


def draw_line(x, y, std, color, label, lens):
    lwidth = 0.6
    plt.plot(x, y, 'k', color=color, marker='o', fillstyle='none',
             linewidth=lwidth, markersize=5, markeredgewidth=lwidth, label=label)
    plt.fill_between(lens, (y-std).clip(0), y + std, alpha=0.2,
                     facecolor=color, linewidth=lwidth, antialiased=True)


def draw_diagram(scobee, our, y_label, lens, thresholds, idx, draw_scobee=True):
    plt.figure()
    if draw_scobee:
        draw_line(lens, scobee.mean(0), scobee.std(0), 'blue', 'Scobee($d_{kl} = 0.1$)', lens)
    for i in idx:
        draw_line(lens, our[i].mean(0), our[i].std(0),
                  colors[i], f'MESC-IRL($\zeta\geq{thresholds[i]}$)', lens)
    plt.legend()
    plt.xlabel('Number of Demonstrations')
    plt.ylabel(y_label)
    plt.grid(axis='both', which='major', ls='--', lw=0.5)


def main():
    Path('./reports/hard/').mkdir(exist_ok=True, parents=True)
    true_trj = load_data("./data/scobee_example_trajectories.json")
    true_data = load_data("./data/scobee_example_data.json")
    scobee = load_data("./data/scobee_results_scobee_example.json")
    thresholds = [0.4, 0.5, 0.6, 0.7]
    idx = [0, 1, 2, 3]
    lens = list(range(1, 10)) + list(range(10, 101, 10))
    n_len = len(lens)
    n_set = 10

    true_cons = get_true_cons(true_data)
    scobee_cons = get_predicted_cons(scobee, n_set, n_len)
    true_demo = true_trj['trajs']
    scobee_demo = scobee['demos']

    s_fp, s_kl = get_stats(true_cons, scobee_cons,
                           true_demo, scobee_demo, n_set, n_len)
    o_fp, o_kl = [0] * len(thresholds), [0] * len(thresholds)
    for i, t in enumerate(thresholds):
        our = load_data(f"./results/hard/our_results_scobee_example_t{t}.json")
        our_cons = get_predicted_cons(our, n_set, n_len)
        our_hard_demo = our['hard_demos']
        o_fp[i], o_kl[i] = get_stats(
            true_cons, our_cons, true_demo, our_hard_demo, n_set, n_len)

    draw_diagram(s_fp, o_fp, 'False Positive Rate', lens,
                 thresholds, idx, draw_scobee=True)
    plt.savefig('./reports/hard/hard_all_fp.pdf')

    draw_diagram(s_kl, o_kl, 'KL-Divergence', lens,
                 thresholds, idx, draw_scobee=True)
    plt.savefig('./reports/hard/hard_all_kl.pdf')

    draw_diagram(s_fp, o_fp, 'False Positive Rate', lens,
                 thresholds, idx, draw_scobee=False)
    plt.savefig('./reports/hard/hard_ours_fp.pdf')

    draw_diagram(s_kl, o_kl, 'KL-Divergence', lens,
                 thresholds, idx, draw_scobee=False)
    plt.savefig('./reports/hard/hard_ours_kl.pdf')

    draw_diagram(s_fp, o_fp, 'False Positive Rate',
                 lens, thresholds, [2], draw_scobee=True)
    plt.savefig('./reports/hard/hard_best_fp.pdf')
    
    draw_diagram(s_kl, o_kl, 'KL-Divergence', lens,
                 thresholds, [2], draw_scobee=True)
    plt.savefig('./reports/hard/hard_best_kl.pdf')


if __name__ == "__main__":
    main()
