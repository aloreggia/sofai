"""
Adopted from https://github.com/qzed/irl-maxent/blob/master/src/solver.py.

Generic solver methods for Markov Decision Processes (MDPs) and methods for
policy computations for GridWorld.
"""

import numpy as np
import scipy.special


def value_iteration(p, reward, discount, eps=1e-3):
    """
    Basic value-iteration algorithm to solve the given MDP.

    Args:
        p: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        reward: The reward signal per state as table
            `[state: Integer, action: Integer] -> reward: Float`.
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The value function as table `[state: Integer] -> value: Float` and
        the q-value function as table `[state: Integer, actiuon: Integer] -> value: Float`
    """
    n_states, _, _ = p.shape
    v = np.zeros(n_states)
    p_t = np.moveaxis(p, 1, 2)

    delta = np.inf
    q = 0
    while delta > eps:      # iterate until convergence
        v_old = v

        # compute state-action values (note: we actually have Q[a, s] here)
        # sum over the destination states
        q = (p_t * (discount * v[None, None, :] + reward)).sum(-1)

        # compute state values
        v = np.max(q.T, axis=0)

        # compute maximum delta
        delta = np.max(np.abs(v_old - v))

    return q, v


def stochastic_policy_from_q_value(world, q_value, w=lambda x: x):
    """
    Compute a stochastic policy from the given q-value function.
    Args:
        world: The `GridWorld` instance for which the the policy should be
            computed.
        q_value: The q_value-function dictating the policy as table
            `[state: Integer, action: Integer] -> value: Float`
        w: A weighting function `(value: Float) -> value: Float` applied to
            all state-action values before normalizing the results, which
            are then used as probabilities. I.e. choosing `x -> 2*x` here
            will cause the preference of suboptimal actions to decrease
            quadratically compared to the preference of the optimal action 
            as we are using a softmax distribution.
    Returns:
        The stochastic policy given the provided arguments as table
        `[state: Integer, action: Integer] -> probability: Float`
        describing a probability distribution p(action | state) of selecting
        an action given a state.
    """
    return scipy.special.softmax(w(q_value), 1)
