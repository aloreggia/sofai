"""
Utilities for plotting.
"""

from itertools import product

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib


def plot_state_values(ax, world, values, start, goal, colored=[], **kwargs):
    """
    Plot the given state values of a GridWorld instance.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the state-values should be plotted.
        values: The state-values to be plotted as table
            `[state: Integer] -> value: Float`.
        start: The start states
        goal: The goal states

        All further key-value arguments will be forwarded to
        `pyplot.imshow`.
    """
    print(kwargs)
    p = ax.imshow(np.reshape(values, (world.size, world.size)),
                  origin='lower', **kwargs)
    ax.axis('off')

    colored = [(world.state_index_to_point(s), c) for s, c in colored]

    for i in range(0, world.size + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, world.size - 0.5], c='black', lw=0.7)
        ax.plot([-0.5, world.size - 0.5],
                [i - 0.5, i - 0.5], c='black', lw=0.7)

    for (x, y), c in colored:
        ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color=c, lw=3)
        ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color=c, lw=3)
        ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color=c, lw=3)
        ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color=c, lw=3)

    for s in start:
        x, y = world.state_index_to_point(s)
        ax.text(x - 0.3, y - 0.2, '$s_0$')

    for s in goal:
        x, y = world.state_index_to_point(s)
        ax.text(x - 0.3, y - 0.2, '$s_G$')

    return p


def plot_trajectory(ax, world, trajectory, **kwargs):
    """
    Plot a trajectory as line.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the trajectory should be plotted.
        trajectory: The `Trajectory` object to be plotted.

        All further key-value arguments will be forwarded to
        `pyplot.tripcolor`.

    """
    xy = [world.state_index_to_point(s) for s, _, _ in trajectory.transitions()] 

    if len(trajectory.transitions()):
        xy += [world.state_index_to_point(trajectory.transitions()[-1][-1])]
    x, y = zip(*xy)

    return ax.plot(x, y, **kwargs)


def plot_action_rewards(ax, actions_rewards, **kwargs):
    values = np.zeros((3, 3))
    for a, r in actions_rewards.items():
        x = 1 + a.x
        y = 1 + a.y
        values[y, x] = r
        ax.arrow(1 + 2 * a.x/10, 1 + 2 * a.y/10, 7 * a.x / 10,
                 7 * a.y / 10, head_width=0.1, linewidth=2)
    p = ax.imshow(values, origin='lower', **kwargs)
    ax.axis('off')

    for i in range(0, 4):
        ax.plot([i - 0.5, i - 0.5], [-0.5, 3 - 0.5], c='black', lw=0.7)
        ax.plot([-0.5, 3 - 0.5], [i - 0.5, i - 0.5], c='black', lw=0.7)
    return p


def plot_colors(ax, vals, **kwargs):
    p = plt.imshow(vals.reshape(-1, 1), origin='lower', **kwargs)
    cmap = p.cmap
    p = ax.imshow(vals.reshape(-1, 1), origin='lower', **kwargs)
    ax.axis('off')

    size = 1
    ax.text(0.8 * size, -0.1 * size, 'No color', size=8, fontweight='bold')
    ax.text(0.8 * size, 0.9 * size, 'Blue', size=8,
            color='blue', fontweight='bold')
    ax.text(0.8 * size, 1.9 * size, 'Green', size=8,
            color='green', fontweight='bold')

    eps = 0.05
    ax.add_patch(Rectangle((-0.5, 0.5), size, size-eps,
                           edgecolor='blue', facecolor=cmap(p.norm(vals[1])), lw=3))
    ax.add_patch(Rectangle((-0.5, 1.5 + eps), size, size - 2*eps,
                           edgecolor='green', facecolor=cmap(p.norm(vals[2])), lw=3))

    return p
