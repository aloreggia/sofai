#!/usr/bin/env python

from logging import debug, root, DEBUG
from collections import namedtuple, defaultdict
import math
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from max_ent.algorithms.icrl import icrl
from max_ent.gridworld import Directions
import max_ent.gridworld.feature as F
import max_ent.examples.grid_plot as P
from max_ent.algorithms.gridworld_icrl import generate_random_trajectories, learn_constraints, setup_mdp, \
    generate_mdft_trajectories, generate_trajectories, MDP, generate_weighted_average_trajectories, generate_random_trajectories, generate_greedy_trajectories


Config = namedtuple('Config', ['mdp', 'state_penalties',
                               'action_penalties', 'color_penalties', 'blue', 'green'])


def plot_world(title, mdp, state_rewards, action_rewards, color_rewards,
               demo, blue_states, green_states, vmin=None, vmax=None):

    cm = plt.cm.afmhot
    fsize = (4.5, 3)
    fig = plt.figure(num=title, figsize=fsize)
    spec = gridspec.GridSpec(ncols=12, nrows=2, figure=fig)
    colored = [(s, 'blue') for s in blue_states]
    colored += [(s, 'green')for s in green_states]

    ax = fig.add_subplot(spec[:, :8])
    p = P.plot_state_values(ax, mdp.world, state_rewards, mdp.start,
                            mdp.terminal, colored, vmin=vmin, vmax=vmax, cmap=cm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(p, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")

    for t in demo.trajectories:
        P.plot_trajectory(ax, mdp.world, t, lw=4,
                          color='red', alpha=min(1, 2/len(demo.trajectories)))

    ax = fig.add_subplot(spec[0, 9:12])
    P.plot_action_rewards(ax, action_rewards, vmin=vmin, vmax=vmax, cmap=cm)

    ax = fig.add_subplot(spec[1, 10])
    P.plot_colors(ax, color_rewards, vmin=vmin, vmax=vmax, cmap=cm)
    plt.draw()
    return fig


def plot_trajectory_comparison(title, mdp, state_rewards, action_rewards, color_rewards,
                               demo1, demo2, blue_states, green_states, vmin=None, vmax=None):

    fsize = (4.5, 3)
    fig = plt.figure(num=title, figsize=fsize)
    spec = gridspec.GridSpec(ncols=12, nrows=2, figure=fig)
    colored = [(s, 'blue') for s in blue_states]
    colored += [(s, 'green')for s in green_states]

    ax = fig.add_subplot(spec[:, :8])
    p = P.plot_state_values(ax, mdp.world, state_rewards, mdp.start,
                            mdp.terminal, colored, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(p, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")

    for t in demo1.trajectories:
        P.plot_trajectory(ax, mdp.world, t, lw=4,
                          color='blue', alpha=0.025)

    for t in demo2.trajectories:
        P.plot_trajectory(ax, mdp.world, t, lw=4,
                          color='red', alpha=0.025)

    ax = fig.add_subplot(spec[0, 9:12])
    P.plot_action_rewards(ax, action_rewards, vmin=vmin, vmax=vmax)

    ax = fig.add_subplot(spec[1, 10])
    P.plot_colors(ax, color_rewards, vmin=vmin, vmax=vmax)
    plt.draw()
    return fig


def config_world(blue, green, constrained_states, constrained_actions, constrained_colors, goal,
                 penalty=-50, start=[0], p_slip=0.1, dist_penalty=True, default_reward=-4):
    size = 9
    action_penalty, state_penalty, color_penalty = penalty, penalty, penalty
    goal_r = 10

    # set-up the mdp
    sf = F.DistinationStateFeature(size**2)
    af = F.DirectionFeature(Directions.ALL_DIRECTIONS)
    cf = F.ColorFeature(['No Color', 'Blue', 'Green'], 0)
    cf.set_states_color(blue, 'Blue')
    cf.set_states_color(green, 'Green')

    # Destination state penalties
    constraints = []
    sp = np.zeros(size**2)
    sp[goal] = goal_r
    for s in constrained_states:
        constraints.append((sf, s, state_penalty))
        sp[s] = state_penalty

    # Action penalties
    ap = {a: default_reward * (np.sqrt(a.x**2 + a.y**2) if dist_penalty else 1)
          for a in Directions.ALL_DIRECTIONS}
    for a in ap:
        if a in constrained_actions:
            ap[a] = action_penalty

        constraints.append((af, a.idx, ap[a]))

    # Color
    cp = np.zeros(3)
    for c in constrained_colors:
        constraints.append((cf, c, color_penalty))
        cp[c] = color_penalty

    feature_list = [sf, af, cf]
    goal = goal if isinstance(goal, list) or isinstance(
        goal, np.ndarray) else [goal]
    start = start if isinstance(start, list) or isinstance(
        start, np.ndarray) else [start]
    mdp = setup_mdp(size, feature_list, constraints, terminal=goal,
                    terminal_reward=goal_r, start=start, p_slip=p_slip)

    return Config(mdp, sp, ap, cp, blue, green)


def generate_random_config(min_dist_start_goal=5, p_slip=0.1, penalty=-50, dist_penalty=False):
    n_const = 6
    # generate the list of non-constrained states
    list_available = [x for x in range(81)]

    blue = np.random.choice(list_available, n_const,
                            replace=False)  # blue states
    # remove blue states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, blue)

    green = np.random.choice(list_available, n_const,
                             replace=False)  # green states
    # remove green states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, green)

    cs = np.random.choice(list_available, n_const,
                          replace=False)  # constrained states
    # remove constrained states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, cs)

    random_ca = np.random.choice(8, 2, replace=False)
    ca = [Directions.ALL_DIRECTIONS[d] for d in random_ca]

    cc = [1, 2]

    start = random.choice(list_available)
    # generate terminal state from the list of non-constrained states
    goal = random.choice(list_available)
    while (start % 9 - goal % 9)**2 + (start/9 - goal/9)**2 < min_dist_start_goal**2:
        start = random.choice(list_available)
        goal = random.choice(list_available)

    n_cfg = config_world(blue, green, [], [], [], goal, dist_penalty=dist_penalty,
                         start=start, p_slip=p_slip, penalty=penalty)
    c_cfg = config_world(blue, green, cs, ca, cc, goal, dist_penalty=dist_penalty,
                         start=start, p_slip=p_slip, penalty=penalty)

    return n_cfg, c_cfg


def main():
    np.random.seed(123)
    goal = 8
    blue = [4, 13, 22]  # blue states
    green = [58, 67, 76]  # green states
    cs = [31, 39, 41, 47, 51]  # constrained states
    ca = [Directions.UP_LEFT, Directions.UP_RIGHT]  # constrained actions
    cc = [1, 2]  # constrained colors

    n_cfg = config_world(blue, green, [], [], [], goal)
    c_cfg = config_world(blue, green, cs, ca, cc, goal)
    n, c = n_cfg.mdp, c_cfg.mdp

    # Generate demonstrations and plot the world
    demo = generate_trajectories(n.world, n.reward, n.start, n.terminal)
    vmin = c_cfg.state_penalties.min()
    vmax = c_cfg.state_penalties.max()
    plot_world('Nominal', n, n_cfg.state_penalties,
               n_cfg.action_penalties, n_cfg.color_penalties,
               demo, n_cfg.blue, n_cfg.green, vmin=vmin, vmax=vmax)

    demo = generate_trajectories(c.world, c.reward, c.start, c.terminal)
    plot_world('Original Constrained', c, c_cfg.state_penalties,
               c_cfg.action_penalties, c_cfg.color_penalties,
               demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)
    # plt.show()

    # Learn the constraints
    result = learn_constraints(
        n.reward, c.world, c.terminal, demo.trajectories)

    print("learning finished!")

    learned = MDP(c.world, result.reward, c.terminal, c.start)
    demo = generate_trajectories(
        learned.world, learned.reward, learned.start, learned.terminal)
    plot_world('Learned Constrained', learned, result.state_weights, result.action_weights,
               result.color_weights, demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)

    mdf_demo = generate_mdft_trajectories(
        n.world, n.reward, result.reward, n.start, n.terminal, [0.1, 0.9])

    plot_world('Learned Constrained - MDFT trajectories', learned, result.state_weights, result.action_weights,
               result.color_weights, mdf_demo, c_cfg.blue, c_cfg.green, vmin=vmin, vmax=vmax)

    plt.show()
    debug('Done!')


if __name__ == '__main__':
    main()
