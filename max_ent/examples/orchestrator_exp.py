from max_ent.algorithms.gridworld_icrl import generate_optimal_trajectories
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import random
import pickle
from scipy import stats

import max_ent.examples.grid_9_by_9 as G
from max_ent.utility.support import generate_constraints
from max_ent.gridworld import Directions

colors = ['purple', 'red', 'orange', 'green', 'blue', 'yellow']
N_TRAJ = 100


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


def create_world(blue, green, cs=[], ca=[], cc=[], start=0, goal=8):
    n_cfg = G.config_world(blue, green, cs, ca, cc, goal, start=start)
    n = n_cfg.mdp

    # Generate demonstrations and plot the world
    demo = G.generate_trajectories(
        n.world, n.reward, n.start, n.terminal, n_trajectories=1)
    if not demo:
        return None, None

    return n, n_cfg


def learn_random_worlds(n_tests):
    results = []
    while len(results) < n_tests:
        blue, green, cs, ca, start, goal = generate_constraints(9)
        n, _ = create_world(blue, green, start=start, goal=goal)

        cc = [1, 2]
        c, _ = create_world(blue, green, cs, ca, cc, start=start, goal=goal)

        # CHECK WHETHER STATE AND GOAL ARE REACHABLE - IF NOT SKIP THE GRID AND GENERATE A NEW ONE
        if c == None:
            continue

        print(f'Learning world #{len(results) + 1}')

        demo_n = G.generate_trajectories(
            n.world, n.reward, n.start, n.terminal, n_trajectories=N_TRAJ)
        demo_c = G.generate_trajectories(
            c.world, c.reward, c.start, c.terminal, n_trajectories=N_TRAJ)

        learned_params = G.learn_constraints(
            n.reward, c.world, c.terminal, demo_c.trajectories)
        demo_l = G.generate_trajectories(
            c.world, learned_params.reward, c.start, c.terminal, n_trajectories=N_TRAJ)

        results.append({
            'start': start,
            'goal': goal,
            'learned_params': learned_params,
            'demo_n': demo_n.trajectories,
            'demo_c': demo_c.trajectories,
            'demo_l': demo_l.trajectories,
            'constraints': {'blue': blue, 'green': green, 'cs': cs, 'ca_idx': [a.idx for a in ca], 'ca': ca}
        })
    return results


def get_worlds(d):
    const = d['constraints']
    n, _ = create_world(const['blue'], const['green'], start=d['start'],
                        goal=d['goal'])

    l = G.MDP(n.world, d['learned_params'].reward, n.terminal, n.start)

    return n, l


def get_traj_stats(traj, reward, constraints):
    avg_length = 0
    avg_pen = 0
    cs, cc, ca = 0, 0, 0
    n = len(traj)
    for t in traj:
        for s, a, s_ in t.transitions():
            avg_length += 1
            avg_pen += reward[s, a, s_]
            if s in constraints['cs']:
                cs += 1

            if a in constraints['ca_idx']:
                ca += 1

            if s in (constraints['blue'] + constraints['green']):
                cc += 1

    avg_length /= n
    avg_pen /= n
    cs /= n
    ca /= n
    cc /= n
    violations = cs + ca + cc

    return avg_length, avg_pen, violations


def get_stats(demo, traj, reward, constraints, len_baseline, pen_baseline,):
    avg_length, avg_pen, avg_vio = get_traj_stats(traj, reward, constraints)
    return avg_length / len_baseline, avg_pen / pen_baseline, avg_vio, jsd(demo, traj)


def get_length_baselines(demo_n, demo_l):
    n_lens = [len(t.transitions()) for t in demo_n]
    c_lens = [len(t.transitions()) for t in demo_l]
    min_length = min(n_lens)
    avg_nominal_length = sum(n_lens) / len(n_lens)
    avg_constrained_length = sum(c_lens) / len(c_lens)

    return min_length, avg_nominal_length, avg_constrained_length


def get_penalty_baselines(demo_n, demo_l, demo_g, reward):
    def r(t):
        return sum([reward[x] for x in t.transitions()])
    p_n = sum([r(t) for t in demo_n]) / len(demo_n)
    p_l = sum([r(t) for t in demo_l]) / len(demo_l)
    p_g = sum([r(t) for t in demo_g]) / len(demo_g)

    return p_n, p_l, p_g


def get_violation_baselines(demo_n, demo_l, demo_g, constraints):
    color_const = constraints['blue'].tolist() + constraints['green'].tolist()
    def v(t):
        cs, cc, ca = 0, 0, 0
        for s, a, _ in t.transitions():
            if s in constraints['cs']:
                cs += 1

            if a in constraints['ca_idx']:
                ca += 1

            if s in color_const:
                cc += 1
        return cs + ca + cc
    v_n = sum([v(t) for t in demo_n]) / len(demo_n)
    v_l = sum([v(t) for t in demo_l]) / len(demo_l)
    v_g = sum([v(t) for t in demo_g]) / len(demo_g)

    return v_n, v_l, v_g


def get_orchestrator_results(learned):
    wa = np.zeros((len(learned), 11, 4))
    mdft = np.zeros((len(learned), 11, 4))
    avg_min_len, avg_n_len, avg_c_len = 0, 0, 0
    avg_n_pen, avg_c_pen, avg_g_pen = 0, 0, 0
    avg_n_v, avg_c_v, avg_g_v = 0, 0, 0
    n_tests = len(learned)
    for i, d in enumerate(learned):
        print(f'Processing world #{i+1} ...')
        n, l = get_worlds(d)
        demo = d['demo_c']
        aml, anl, acl = get_length_baselines(d['demo_n'], d['demo_l'])
        avg_min_len += aml
        avg_n_len += anl / aml
        avg_c_len += acl / aml

        demo_g = G.generate_greedy_trajectories(n.world, n.reward, d['learned_params'].reward,
                                                n.start, n.terminal,
                                                n_trajectories=N_TRAJ).trajectories

        p_n, p_l, p_g = get_penalty_baselines(
            d['demo_n'], d['demo_l'], demo_g, l.reward)
        avg_n_pen += p_n / p_l
        avg_c_pen += p_l
        avg_g_pen += p_g / p_l

        v_n, v_l, v_g = get_violation_baselines(
            d['demo_n'], d['demo_l'], demo_g, d['constraints'])
        avg_n_v += v_n
        avg_c_v += v_l
        avg_g_v += v_g

        for j in range(11):
            w = [(j)/10, 1 - (j)/10]
            wa_traj = G.generate_weighted_average_trajectories(n.world, n.reward, d['learned_params'].reward,
                                                               n.start, n.terminal, w,
                                                               n_trajectories=N_TRAJ).trajectories
            wa[i, j] = get_stats(demo, wa_traj, l.reward, d['constraints'], aml, p_l)

            mdft_traj = G.generate_mdft_trajectories(n.world, n.reward, d['learned_params'].reward,
                                                     n.start, n.terminal, w,
                                                     n_trajectories=N_TRAJ).trajectories
            mdft[i, j] = get_stats(demo, mdft_traj, l.reward, d['constraints'], aml, p_l)


    avg_min_len /= n_tests
    avg_n_len /= n_tests
    avg_c_len /= n_tests

    avg_n_pen /= n_tests
    avg_c_pen /= n_tests
    avg_g_pen /= n_tests

    avg_n_v /= n_tests
    avg_c_v /= n_tests
    avg_g_v /= n_tests

    return wa, mdft, avg_min_len, avg_n_len, avg_c_len, avg_n_pen, \
        avg_c_pen, avg_g_pen, avg_n_v, avg_c_v, avg_g_v


def draw_metric(wa, mdft, lines, y_label,  labels, filename):

    y = np.stack([wa.mean(0), mdft.mean(0)])
    sem = np.stack([stats.sem(wa, 0), stats.sem(mdft, 0)])
    x = list(range(1, 12))
    plt.figure(figsize=(12, 7))
    lwidth = 0.6
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :], 'k', color=colors[i], marker='o', fillstyle='none',
                 linewidth=lwidth, markersize=5, markeredgewidth=lwidth, label=labels[i])
        plt.fill_between(x, (y[i, :] - sem[i, :]), y[i, :] + sem[i, :], alpha=0.2,
                         facecolor=colors[i], linewidth=lwidth, antialiased=True)
    i = y.shape[0]
    for l in lines:
        plt.axhline(y=l, color=colors[i], ls='--', label=labels[i], lw=1)
        i += 1

    xlabels = [f'({w/10:0.1f}, {1 - w/10:0.1f})' for w in range(11)]
    plt.xticks(x, labels=xlabels)
    plt.legend(fontsize=14)
    plt.xlabel('$(\mathbf{w}_n, \mathbf{w}_c)$', fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    plt.grid(axis='both', which='major', ls='--', lw=0.5)
    plt.savefig(f'./reports/orchestrator/orchestrator_{filename}.pdf')
    plt.close()


def main():
    n_tests = 100
    learn = False
    random.seed(123)
    np.random.seed(123)
    if learn:
        learned = learn_random_worlds(n_tests)
        with open(f'results/orchestrator/learned_mdps_{n_tests}.pkl', 'wb') as f:
            pickle.dump(learned, f)
    else:
        with open(f'results/orchestrator/learned_mdps_{n_tests}.pkl', 'rb') as f:
            learned = pickle.load(f)
        wa, mdft, aml, anl, acl, anp, acp, agp, anv, acv, agv = get_orchestrator_results(
            learned)

        draw_metric(wa[:, :, 0], mdft[:, :, 0], [1, anl, acl],
                    'Avg Norm. Length', ['WA', 'MDFT', 'Shortest Path', 'Nominal', 'Constrained'], 'length')

        draw_metric(wa[:, :, 1], mdft[:, :, 1], [anp, 1, agp],
                    'Avg Norm. Penalty', ['WA', 'MDFT', 'Nominal', 'Constrained', 'Greedy'], 'penalty')

        draw_metric(wa[:, :, 2], mdft[:, :, 2], [anv, acv, agv],
                    'Avg Num Violated Constraints', ['WA', 'MDFT', 'Nominal', 'Constrained', 'Greedy'], 'violations')

        draw_metric(wa[:, :, 3], mdft[:, :, 3], [],
                    'Avg JS-Divergence', ['WA', 'MDFT'], 'jsd')
        print('.')


if __name__ == "__main__":
    main()
