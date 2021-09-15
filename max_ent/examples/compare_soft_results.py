from collections import defaultdict, namedtuple
from max_ent.examples.compare_hard_results import main
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP
import json
import pickle
from scipy import stats
from pathlib import Path

PlotSetting = namedtuple('PlotSetting', [
    'filter', 'type', 'name', 'p_thresholds', 'count', 'mae'])

lens = list(range(1, 10)) + list(range(10, 101, 10))
colors = ['purple', 'red', 'orange', 'green']

plots = [
    # Like false positive
    PlotSetting('zero', 'over', 'fp', True, True, False),
    PlotSetting('non-zero', 'under', 'fn', True,
                True, False),  # Like false negative
]
p_thresholds = [0.01, 0.05, 0.1, 0.2]


def dist(demo):
    dist = np.ones((81, 8)) * 1e-6
    for t in demo:
        for s, a, _ in t.transitions():
            dist[s, a] += 1
    return dist/dist.sum().reshape(-1, 1)


def jsd(x, y):
    def kl(p, q):
        kl = p * np.log2(p/q)
        return kl.sum()
    p = dist(x)
    q = dist(y)
    m = (p + q) / 2
    return (kl(p, m) + kl(q, m))/2


def mae(true, learned, p_true, p_learned, filter, type, eps=0.01):
    cond = get_condition(true, filter, type, p_true, p_learned, eps)
    if cond.sum() == 0:
        return 0
    return np.abs(true[cond] - learned[cond]).sum() / len(true)


def count(true, p_true, p_learned, filter, type, eps=0.01):
    cond = get_condition(true, filter, type, p_true, p_learned, eps)
    return cond.sum() / len(true)


def get_condition(true, filter, type, p_true, p_learned, eps):
    t = np.ones_like(true).astype(np.bool)
    f = np.ones_like(true).astype(np.bool)
    if filter == 'zero':
        f = p_true == 0
    if type == 'over':
        f = p_true != 0

    if type == 'under':
        t = (p_true - p_learned) > eps
    if type == 'over':
        t = (p_learned - p_true) > eps

    return f & t


def get_probs(nominal_reward, result):
    ps, pa, pc = convert_constraints_to_probs(nominal_reward, result)
    pa = np.array([pa[a] for a in Directions.ALL_DIRECTIONS])
    p = np.concatenate([ps, pa, pc])
    return p


def get_results(d, r, p_slip):
    Path('./reports/soft/').mkdir(exist_ok=True, parents=True)
    np.random.seed(123)
    n, c, cs, ca_idx, cc = get_configs(d, p_slip)

    omega = np.zeros(81 + 8 + 3)
    az_result = convert_to_icrl_result(omega, n.world, n.reward)

    omega[cs] = -np.array(d['state_reward'])[cs]
    omega[81 + ca_idx] = -np.array(d['action_reward'])[ca_idx]
    omega[89 + cc] = -np.array(d['color_reward'])[cc]

    true_result = convert_to_icrl_result(omega, n.world, c.reward)
    p_true = get_probs(n.reward, true_result)

    demo = generate_trajectories(
        c.world, c.reward, c.start, c.terminal, n_trajectories=100)
    az_demo = generate_trajectories(
        c.world, az_result.reward, n.start, n.terminal, n_trajectories=100)
    az_jsd = jsd(demo.trajectories, az_demo.trajectories)

    N = len(r)
    plot_list = {k: {
        'mae': [np.zeros((N, len(lens))) for _ in p_thresholds],
        'count': [np.zeros((N, len(lens))) for _ in p_thresholds],
    } for k in plots}
    jsd_list = np.zeros((N, len(lens)))

    for i in range(N):
        for j, result in enumerate(r[i]):
            demo = generate_trajectories(
                c.world, c.reward, c.start, c.terminal, n_trajectories=100)

            result = convert_to_icrl_result(
                result['omega'], n.world, result['reward'])
            learned = MDP(n.world, result.reward, n.terminal, n.start)
            learned_demo = generate_trajectories(
                n.world, learned.reward, n.start, n.terminal, n_trajectories=100)

            p_learned = get_probs(n.reward, result)
            jsd_list[i, j] = jsd(demo.trajectories, learned_demo.trajectories)
            for plt_s in plot_list:
                plt_r = plot_list[plt_s]
                m = plt_r['mae']
                cnt = plt_r['count']
                T = p_thresholds if plt_s.p_thresholds else [0]
                for k, t in enumerate(T):
                    m[k][i, j] = mae(true_result.omega,
                                     result.omega, p_true, p_learned, plt_s.filter, plt_s.type, eps=t)
                    cnt[k][i, j] = count(true_result.omega,
                                         p_true, p_learned, plt_s.filter, plt_s.type, eps=t)

    return jsd_list, plot_list, az_jsd


def get_configs(d, p_slip):
    goal = d['goal']
    blue = d['blue']
    green = d['green']
    cs = np.argwhere(np.array(d['state_reward']) == -50).ravel()
    ca_idx = np.argwhere(np.array(d['action_reward']) == -50).ravel()
    ca = [Directions.ALL_DIRECTIONS[d] for d in ca_idx]
    cc = np.argwhere(np.array(d['color_reward']) == -50).ravel()

    n_cfg = config_world(blue, green, [], [], [], goal,
                         start=d['start'], p_slip=p_slip)
    c_cfg = config_world(blue, green, cs, ca, cc, goal,
                         start=d['start'], p_slip=p_slip)
    n = n_cfg.mdp
    c = c_cfg.mdp
    return n, c, cs, ca_idx, cc


def convert_to_icrl_result(omega, world, reward):
    omega_action = {a: -omega[world.n_states + i]
                    for i, a in enumerate(world.actions)}
    omega_state = -omega[:world.n_states].copy()
    omage_color = -omega[world.n_states + world.n_actions:].copy()
    true_result = ICRL_Result(
        omega.copy(), reward.copy(), omega_state, omega_action, omage_color)
    return true_result


def draw_metric(x, y, sem, y_label, labels, filename, az=None):
    plt.figure()
    lwidth = 0.6
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :], 'k', color=colors[i], marker='o', fillstyle='none',
                 linewidth=lwidth, markersize=5, markeredgewidth=lwidth, label=labels[i])
        plt.fill_between(lens, (y[i, :] - sem[i, :]).clip(0), y[i, :] + sem[i, :], alpha=0.2,
                         facecolor=colors[i], linewidth=lwidth, antialiased=True)
    if len(labels) > 1:
        plt.legend()

    if az:
        plt.axhline(y=az, color='r', ls='--', label='no constraint prediction')
        plt.legend()


    plt.xlabel('Number of Demonstrations')
    plt.ylabel(y_label)

    plt.grid(axis='both', which='major', ls='--', lw=0.5)
    plt.savefig(f'./reports/soft/{filename}_soft.pdf')
    plt.close()


def draw_result(jsd_list, az_jsd, plot_list, exp_name):

    draw_metric(lens, jsd_list['mean'].reshape(1, -1),
                np.sqrt(jsd_list['var']).reshape(1, -1), 'JS-Divergence',
                ['MESC-IRL'], f'{exp_name}_jsd', az_jsd)

    for k in plot_list:
        p = plot_list[k]
        for metric in ['mae', 'count']:
            if not k.mae and metric == 'mae':
                continue
            if not k.count and metric == 'count':
                continue

            m = p[metric]
            mean_ = np.array(
                m['mean']) if k.p_thresholds else m['mean'][0].reshape(1, -1)
            var_ = np.array(
                m['var']) if k.p_thresholds else m['var'][0].reshape(1, -1)
            sem_ = np.sqrt(var_)

            labels = ['']
            title = ''
            if k.p_thresholds:
                l = '' if metric == 'count' else 'Mean Absolute Error'
                if k.type == 'over':
                    l = '$\chi = $ '
                    if metric == 'count':
                        title = 'Soft False Positive Rate'

                if k.type == 'under':
                    l = '$\chi = $ '
                    if metric == 'count':
                        title = 'Soft False Negative Rate'
                labels = [l + f'${t}$' for t in p_thresholds]

            draw_metric(lens, mean_, sem_,
                        title, labels,
                        f'{exp_name}_{metric}_{k.type}')


def draw_reports(name, deter):
    with open(f'data/{name}.json') as f:
        data = json.load(f)

    p_slip = 0 if deter else 0.1
    name += '_deter' if deter else '_non_deter'
    with open(f'results/soft/{name}.pkl', 'rb') as f:
        results = pickle.load(f)

    jsd_list = {'mean': 0, 'var': 0}
    plt_list = {}
    n_games = len(results)
    n_thresholds = len(p_thresholds)
    az_jsd = 0

    for i, r in enumerate(results):
        jsd_, plt_result, az_ = get_results(data[i], r, p_slip)
        jsd_list['mean'] += jsd_.mean(0) / n_games
        az_jsd += az_ / n_games
        # convert to variance for pooled st calculation. Equal sample sizes
        # st_pooling**2 = (st_1**2 + ... + st_n**2)/k, k is the number of samples
        jsd_list['var'] += stats.sem(jsd_, 0) ** 2 / n_games

        for k in plt_result:
            r = plt_result[k]
            if k not in plt_list:
                plt_list[k] = {
                    'mae': {'mean': [0] * n_thresholds, 'var': [0] * n_thresholds},
                    'count': {'mean': [0] * n_thresholds, 'var': [0] * n_thresholds},
                }
            for i, t in enumerate(p_thresholds if k.p_thresholds else [0]):
                for metric in plt_list[k]:
                    plt_list[k][metric]['mean'][i] += r[metric][i].mean(
                        0) / n_games
                    plt_list[k][metric]['var'][i] += stats.sem(
                        r[metric][i]) ** 2 / n_games

    draw_result(jsd_list, az_jsd, plt_list, name)


def main():
    draw_reports('scobee_example_data', True)
    draw_reports('scobee_example_data', False)
    draw_reports('random_data_10', True)
    draw_reports('random_data_10', False)
    print('Done!')


if __name__ == '__main__':
    main()
