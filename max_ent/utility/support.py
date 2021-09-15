from max_ent.gridworld import Directions
import max_ent.examples.grid_9_by_9 as G
import seaborn as sns
import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.spatial import distance
import random
import pickle
from scipy import stats

# allow us to re-use the framework from the src directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../')))


def conf_interval(array):
    mean, sigma = np.mean(array), np.std(array)
    N = len(array)
    conf_int_a = stats.norm.interval(0.7, loc=mean, scale=sigma/math.sqrt(N))
    #print(f"N: {N} \t Mean: {mean} \t Sigma: {sigma} \t conf_int: {conf_int_a}")
    return (conf_int_a[1] - conf_int_a[0])/2


def create_world(title, blue, green, cs=[], ca=[], cc=[], start=0, goal=8, vmin=-50,
                 vmax=10, check=False, draw=True, n_trajectories=200):
    n_cfg = G.config_world(blue, green, cs, ca, cc, goal, start=start)
    n = n_cfg.mdp

    # Generate demonstrations and plot the world
    if check:
        demo = G.generate_trajectories(
            n.world, n.reward, n.start, n.terminal, n_trajectories=1)
        if not demo:
            return None, None, None, None  # CHECK WHETHER START AND GOAL ARE REACHABLE

    demo = G.generate_trajectories(
        n.world, n.reward, n.start, n.terminal, n_trajectories=n_trajectories)
    if draw:
        fig = G.plot_world(title, n, n_cfg.state_penalties,
                           n_cfg.action_penalties, n_cfg.color_penalties,
                           demo, n_cfg.blue, n_cfg.green, vmin=vmin, vmax=vmax)
    else:
        fig = None
    return n, n_cfg, demo, fig


def total_reward(trajectory, grid, grid_n, constraints):
    #grid = world.mdp
    #grid_n =nominal.mdp
    reward = 0
    reward_n = 0
    count_cs = 0
    count_ca = 0
    count_cb = 0
    count_cg = 0
    for state in trajectory.transitions():
        # check for action constraints violation
        reward += grid.reward[state]
        reward_n += grid_n.reward[state]
        for constraint in constraints['ca']:
            if (state[1] == constraint.idx):
                count_ca += 1

        # check for color constraints violation
        for constraint in constraints['blue']:
            if (state[0] == constraint):
                count_cb += 1

        # check for color constraints violation
        for constraint in constraints['green']:
            if (state[0] == constraint):
                count_cg += 1

        # check for state constraints violation
        for constraint in constraints['cs']:
            if (state[0] == constraint):
                count_cs += 1

    return reward, reward_n, count_cs, count_ca, count_cb, count_cg


# calculate the kl divergence
def kl_divergence(p, q):
    p = np.reshape(p, (-1, 1))
    q = np.reshape(q, (-1, 1))
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))

# calculate the js divergence


def js_divergence(p, q):
    p = np.reshape(p, (-1, 1))
    q = np.reshape(q, (-1, 1))
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# count how many times a state is visited, and compute the average length of the trajectories
# add nominal world -> compute the average nominal reward for the constrained trajectory


def count_states(trajectories, grid, nominal, constraints):
    #grid = world.mdp
    count_matrix = np.ones((9, 9, 8, 9, 9)) * 1e-10
    avg_length = 0.0
    avg_reward = 0.0
    avg_reward_n = 0.0
    avg_violated = 0.0
    avg_cs = 0.0
    avg_ca = 0.0
    avg_cb = 0.0
    avg_cg = 0.0
    n = len(trajectories)
    for trajectory in trajectories:
        avg_length += len(trajectory.transitions())
        # print(trajectory)
        # print(list(trajectory.transitions()))
        reward, reward_n, count_cs, count_ca, count_cb, count_cg = total_reward(
            trajectory, grid, nominal, constraints)
        avg_reward += reward
        avg_reward_n += reward_n
        avg_violated += (count_cs + count_ca + count_cb + count_cg)
        avg_cs += count_cs
        avg_ca += count_ca
        avg_cb += count_cb
        avg_cg += count_cg
        for transition in trajectory.transitions():
            # print(state)
            state_s = transition[0]
            action = transition[1]
            state_t = transition[2]
            count_matrix[grid.world.state_index_to_point(
                state_s)][action][grid.world.state_index_to_point(state_t)] += 1

    return count_matrix / np.sum(count_matrix), avg_length / n, avg_reward / n, avg_reward_n / n, avg_violated/n, (avg_cs/n, avg_ca/n, avg_cb/n, avg_cg/n)


# check for distance between start and terminal states
def generate_constraints(size):

    # generate the list of non-constrained states
    list_available = [x for x in range(size ** 2)]

    blue = np.random.choice(list_available, 6)  # blue states
    # remove blue states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, blue)

    green = np.random.choice(list_available, 6)  # green states
    # remove green states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, green)

    cs = np.random.choice(list_available, 6)  # constrained states
    # remove constrained states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, cs)

    # print(blue)
    # print(green)
    # print(cs)
    # print(list_available)

    random_ca = np.random.choice(8, 2)  # green states
    ca = [Directions.ALL_DIRECTIONS[d] for d in random_ca]
    # print(ca)

    generate = True
    while generate:
        # generate start state from the list of non-constrained states
        start = random.choice(list_available)
        # generate terminal state from the list of non-constrained states
        goal = random.choice(list_available)

        start_x = start % size
        start_y = start // size

        goal_x = goal % size
        goal_y = goal // size

        if abs(start_x-goal_x) > 2 or abs(start_y-goal_y) > 2:
            generate = False

    return blue, green, cs, ca, start, goal


def plot_statistics(df, learned_matrix, nominal_matrix, denominator, save_path, label="avg_norm_length", label_nominal="avg_length", n_tests=100):

    fig = plt.figure(figsize=(12, 7))
    sns.set_style("white")
    g = sns.barplot(x="i", y=label, hue="type", data=df.loc[(
        df['type'] != "constrained") & (df['type'] != "nominal")], palette="autumn", ci=95)
    g.set_xticks(range(11))  # <--- set the ticks first
    g.set_xticklabels(
        [f"({(i)/10:0.1f}, {1 - (i)/10:0.1f})" for i in range(11)])

    #avg_min_nominal_length = df[denominator]

    constrained_avg_norm_length = np.mean(
        [learned_matrix[i][i][label_nominal]/denominator for i in range(n_tests)])
    nominal_avg_norm_length = np.mean(
        [nominal_matrix[i][i][label_nominal]/denominator for i in range(n_tests)])

    #constrained_avg_norm_length= np.mean([learned_matrix[i][i][label_nominal]/learned_matrix[i][i][label_nominal] for i in range(n_tests)])
    #nominal_avg_norm_length= np.mean([nominal_matrix[i][i][label_nominal]/learned_matrix[i][i][label_nominal] for i in range(n_tests)])

    g.axhline(constrained_avg_norm_length, color='r',
              linestyle='--', label="constrained")
    #g.axhline(nominal_avg_norm_length, color='b', linestyle='--', label="nominal")
    g.axhline(1.0, color='b', linestyle='-', label="shortest")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("W(Nominal, Constraints)")
    plt.ylabel("Avg Length")

    plt.show()
    fig.savefig(os.path.join(save_path, f"{label}.png"), bbox_inches='tight')


def compute_statistics(nominal_matrix, constrained_matrix, learned_matrix, mdft_matrix, worlds, n_tests, i, type_m):
    avg_length_mdft = []
    avg_norm_length_mdft = []

    avg_rew_mdft = []

    avg_norm_rew_mdft = []

    avg_vc_mdft = []

    avg_norm_vc_mdft = []

    avg_js_divergence = []
    avg_js_distance = []

    avg_length_nominal = []
    avg_length_constrained = []
    avg_rew_nominal = []
    avg_rew_constrained = []

    avg_vc_nominal = []
    avg_vc_constrained = []

    avg_js_divergence_nominal_mdft = []
    avg_js_divergence_constrained_mdft = []

    avg_js_distance_nominal_mdft = []
    avg_js_distance_constrained_mdft = []

    each_min_nominal_length = []

    for test in range(n_tests):
        l = [trajectory.transitions()
             for trajectory in worlds[test]['demo_n'][0]]
        min_nominal_length = min(map(len, l))
        each_min_nominal_length.append(min_nominal_length)

        n = np.reshape(nominal_matrix[test][test]['temp_matrix'], (-1, 1))
        c = np.reshape(constrained_matrix[test][test]['temp_matrix'], (-1, 1))
        q = np.reshape(mdft_matrix[test][i][i]['temp_matrix'], (-1, 1))

        avg_length_mdft.append(mdft_matrix[test][i][i]['avg_length'])

        # nominal_matrix[test][test]['avg_length'])
        avg_norm_length_mdft.append(
            mdft_matrix[test][i][i]['avg_length']/min_nominal_length)

        avg_rew_mdft.append(mdft_matrix[test][i][i]['avg_reward'])

        avg_norm_rew_mdft.append(
            mdft_matrix[test][i][i]['avg_reward']/learned_matrix[test][test]['avg_reward'])

        avg_vc_mdft.append(mdft_matrix[test][i][i]['avg_violated'])

        avg_norm_vc_mdft.append(
            mdft_matrix[test][i][i]['avg_violated']/learned_matrix[test][test]['avg_violated'])

        # avg_js_divergence.append(js_divergence(p,q))
        # avg_js_distance.append(distance.jensenshannon(p,q))

        avg_length_nominal.append(nominal_matrix[test][test]['avg_length'])
        avg_length_constrained.append(
            constrained_matrix[test][test]['avg_length'])
        avg_rew_nominal.append(nominal_matrix[test][test]['avg_reward'])
        avg_rew_constrained.append(
            constrained_matrix[test][test]['avg_reward'])
        avg_vc_nominal.append(nominal_matrix[test][test]['avg_violated'])
        avg_vc_constrained.append(
            constrained_matrix[test][test]['avg_violated'])

        avg_js_divergence_nominal_mdft.append(js_divergence(q, n)[0])
        avg_js_divergence_constrained_mdft.append(js_divergence(q, c)[0])

        avg_js_distance_nominal_mdft.append(distance.jensenshannon(n, q)[0])
        avg_js_distance_constrained_mdft.append(
            distance.jensenshannon(c, q)[0])

    dict_mdft = {"i": i, "type": type_m, "avg_length": avg_length_mdft, "avg_norm_length": avg_norm_length_mdft, "avg_reward": avg_rew_mdft, "avg_norm_reward": avg_norm_rew_mdft, "avg_vc": avg_vc_mdft, "avg_norm_vc": avg_norm_vc_mdft, "avg_js_dist_nominal": avg_js_distance_nominal_mdft,
                 "avg_js_dist_constrained": avg_js_distance_constrained_mdft, "avg_js_div_nominal": avg_js_divergence_nominal_mdft, "avg_js_div_constrained": avg_js_divergence_constrained_mdft, "avg_min_nominal_length": np.mean(each_min_nominal_length)}

    return dict_mdft


def compute_statistics_grid(nominal_matrix, constrained_matrix, learned_matrix, mdft_matrix, worlds, n_tests, type_m):
    avg_length_mdft = []
    avg_norm_length_mdft = []

    avg_rew_mdft = []

    avg_norm_rew_mdft = []

    avg_vc_mdft = []

    avg_norm_vc_mdft = []

    avg_js_divergence = []
    avg_js_distance = []

    avg_length_nominal = []
    avg_length_constrained = []
    avg_rew_nominal = []
    avg_rew_constrained = []

    avg_vc_nominal = []
    avg_vc_constrained = []

    avg_js_divergence_nominal_mdft = []
    avg_js_divergence_constrained_mdft = []

    avg_js_distance_nominal_mdft = []
    avg_js_distance_constrained_mdft = []

    each_min_nominal_length = []

    for test in range(n_tests):
        l = [trajectory.transitions()
             for trajectory in worlds[test]['demo_n'][0]]
        min_nominal_length = min(map(len, l))
        each_min_nominal_length.append(min_nominal_length)

        n = np.reshape(nominal_matrix[test][test]['temp_matrix'], (-1, 1))
        c = np.reshape(constrained_matrix[test][test]['temp_matrix'], (-1, 1))
        q = np.reshape(mdft_matrix[test][test]['temp_matrix'], (-1, 1))

        avg_length_mdft.append(mdft_matrix[test][test]['avg_length'])

        # nominal_matrix[test][test]['avg_length'])
        avg_norm_length_mdft.append(
            mdft_matrix[test][test]['avg_length']/min_nominal_length)

        avg_rew_mdft.append(mdft_matrix[test][test]['avg_reward'])

        avg_norm_rew_mdft.append(
            mdft_matrix[test][test]['avg_reward']/learned_matrix[test][test]['avg_reward'])

        avg_vc_mdft.append(mdft_matrix[test][test]['avg_violated'])

        avg_norm_vc_mdft.append(
            mdft_matrix[test][test]['avg_violated']/learned_matrix[test][test]['avg_violated'])

        # avg_js_divergence.append(js_divergence(p,q))
        # avg_js_distance.append(distance.jensenshannon(p,q))

        avg_length_nominal.append(nominal_matrix[test][test]['avg_length'])
        avg_length_constrained.append(
            constrained_matrix[test][test]['avg_length'])
        avg_rew_nominal.append(nominal_matrix[test][test]['avg_reward'])
        avg_rew_constrained.append(
            constrained_matrix[test][test]['avg_reward'])
        avg_vc_nominal.append(nominal_matrix[test][test]['avg_violated'])
        avg_vc_constrained.append(
            constrained_matrix[test][test]['avg_violated'])

        avg_js_divergence_nominal_mdft.append(js_divergence(q, n)[0])
        avg_js_divergence_constrained_mdft.append(js_divergence(q, c)[0])

        avg_js_distance_nominal_mdft.append(distance.jensenshannon(n, q)[0])
        avg_js_distance_constrained_mdft.append(
            distance.jensenshannon(c, q)[0])

    dict_mdft = {"type": type_m, "avg_length": avg_length_mdft, "avg_norm_length": avg_norm_length_mdft, "avg_reward": avg_rew_mdft, "avg_norm_reward": avg_norm_rew_mdft, "avg_vc": avg_vc_mdft, "avg_norm_vc": avg_norm_vc_mdft, "avg_js_dist_nominal": avg_js_distance_nominal_mdft,
                 "avg_js_dist_constrained": avg_js_distance_constrained_mdft, "avg_js_div_nominal": avg_js_divergence_nominal_mdft, "avg_js_div_constrained": avg_js_divergence_constrained_mdft, "avg_min_nominal_length": np.mean(each_min_nominal_length)}

    return dict_mdft
