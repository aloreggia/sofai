"""
Adopted from https://github.com/qzed/irl-maxent/blob/master/src/trajectory.py.

Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain
from mdft_nn.mdft import MDFT, get_time_based_dft_dist
from mdft_nn.helpers.distances import hotaling_S


class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.

    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """

    def __init__(self, transitions):
        self._t = list(transitions)

    def transitions(self):
        """
        The transitions of this trajectory.

        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return list(self._t)

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)


def generate_trajectory(world, policy, start, final, max_len=200):
    """
    Generate a single trajectory.

    Args:
        world: The world for which the trajectory should be generated.
        policy: A function (state: Integer) -> (action: Integer) mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index).
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            returned.

    Returns:
        A generated Trajectory instance adhering to the given arguments.
    """

    state = start

    trajectory = []
    trial = 0
    while state not in final:
        if len(trajectory) > max_len:  # Reset and create a new trajectory
            if trial >= 5:
                print('Warning: terminated trajectory generation due to unreachable final state.')
                return Trajectory(trajectory), False    #break
            trajectory = []
            state = start
            trial += 1

        action = policy(state)

        next_s = range(world.n_states)
        next_p = world.p_transition[state, :, action]

        next_state = np.random.choice(next_s, p=next_p)

        trajectory.append((state, action, next_state))
        state = next_state

    return Trajectory(trajectory), True


def generate_trajectories(n, world, policy, start, final, discard_not_feasable=False):
    """
    Generate multiple trajectories.

    Args:
        n: The number of trajectories to generate.
        world: The world for which the trajectories should be generated.
        policy: A function `(state: Integer) -> action: Integer` mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index), a list of starting
            states (with uniform probability), or a list of starting state
            probabilities, mapping each state to a probability. Iff the
            length of the provided list is equal to the number of states, it
            is assumed to be a probability distribution over all states.
            Otherwise it is assumed to be a list containing all starting
            state indices, an individual state is then chosen uniformly.
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            complete.
        discard_not_feasable: Discard trajectories that not reaching the 
            final state(s)

    Returns:
        A generator expression generating `n` `Trajectory` instances
        adhering to the given arguments.
    """
    start_states = np.atleast_1d(start)

    def _generate_one():
        if len(start_states) == world.n_states:
            s = np.random.choice(range(world.n_states), p=start_states)
        else:
            s = np.random.choice(start_states)

        return generate_trajectory(world, policy, s, final)

    list_tr = []
    for _ in range(n):
        tr, reachable = _generate_one()
        if reachable or not discard_not_feasable:
            list_tr.append(tr)
    
    return list_tr

def policy_adapter(policy):
    """
    A policy adapter for deterministic policies.

    Adapts a deterministic policy given as array or map
    `policy[state] -> action` for the trajectory-generation functions.

    Args:
        policy: The policy as map/array
            `policy[state: Integer] -> action: Integer`
            representing the policy function p(state).

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy.
    """
    return lambda state: policy[state]


def stochastic_policy_adapter(policy):
    """
    A policy adapter for stochastic policies.

    Adapts a stochastic policy given as array or map
    `policy[state, action] -> probability` for the trajectory-generation
    functions.

    Args:
        policy: The stochastic policy as map/array
            `policy[state: Integer, action: Integer] -> probability`
            representing the probability distribution p(action | state) of
            an action given a state.

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy, choosing an action randomly based on the distribution
        defined by the given policy.
    """
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])


def mdft_policy_adapter(nominal_q, constrained_q, w=None, delib_t=100):
    def policy(state):
        r = nominal_q[state]
        c = constrained_q[state]
        M = np.concatenate([r[:, None], c[:, None]], 1)
        S = hotaling_S(M, 0.01, 0.01, 2)
        p0 = np.zeros((M.shape[0], 1))
        mdft = MDFT(M, S, w, p0)
        dist = get_time_based_dft_dist(mdft, 1, delib_t)
        return np.argmax(dist)

    return policy

def greedy_policy_adapter(nominal_q, constrained_q):
    def policy(state):
        r = nominal_q[state]
        c = constrained_q[state]
        m = np.maximum(r, c)
        return np.argmax(m)

    return policy

def random_policy_adapter(n_actions):
    def policy(state):
        return np.random.randint(0, n_actions)

    return policy
