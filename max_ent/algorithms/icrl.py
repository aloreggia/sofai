"""
An extension of https://github.com/qzed/irl-maxent/blob/master/src/maxent.py for
soft constrained irl.
"""

import numpy as np


def backward_causal(p_transition, reward, terminal, discount, eps=1e-5):
    n_states, _, n_actions = p_transition.shape

    # set up terminal reward function
    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=np.float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        reward_terminal[terminal] = 0.0

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    p_t = discount * np.moveaxis(p_transition, 1, 2)
    r = (reward * p_t).sum(-1)  # computes the state-action rewards

    delta = np.inf

    while delta > eps:
        v_old = v
        q = r + p_t @ v_old

        v = reward_terminal
        for a in range(n_actions):
            v = _softmax(v, q[:, a])

        delta = np.max(np.abs(v - v_old))

    # compute and return policy
    return np.exp(q - v[:, None])


def forward(p_transition, p_initial, policy, terminal, eps=1e-5):
    # Don't allow transitions from terminal
    p_t = np.moveaxis(p_transition.copy(), 1, 2)
    p_terminal = p_t[terminal, :, :].copy()
    p_t[terminal, :, :] = 0.0

    d = p_initial.sum(1)
    d_total = d

    delta = np.inf
    while delta > eps:
        # state-action expected visitation
        d_sa = d[:, None] * policy
        # for each state s, multiply the expected visitations of all states to s by their probabilities
        # d_s = sum(sa_ev[s_from, a] * p_t[s_from, a, s] for all s_from, a)
        d_ = (d_sa[:, :, None] * p_t).sum((0, 1))

        delta = np.max(np.abs(d - d_))
        d = d_
        d_total += d

    p_t[terminal, :, :] = p_terminal
    # Distribute the visitation stats of satate to their actions
    d_sa = d_total[:, None] * policy

    # Distribute state-action visitations to the next states
    d_transition = d_sa[:, :, None] * p_t

    return d_transition


def initial_probabilities(n_states, n_actions, trajectories):
    initial = np.zeros((n_states, n_actions))
    for t in trajectories:
        s, a, _ = list(t.transitions())[0]
        initial[s, a] += 1
    return initial / len(trajectories)


def ef_from_trajectories(features, trajectories):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :]

    return fe / len(trajectories)


def icrl(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=100, max_iter=10000, max_penalty=200):

    n_states, n_actions, _, n_features = features.shape

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # compute static properties from trajectories
    e_features = ef_from_trajectories(features, trajectories)
    p_initial = initial_probabilities(n_states, n_actions, trajectories)
    nominal_rewards = np.array(nominal_rewards)

    omega = init(n_features)
    delta = mean_error = np.inf

    optim.reset(omega)
    epoch = 0
    best = None
    best_error = 100000
    while epoch <= burnout or (delta > eps and mean_error > eps_error and epoch < max_iter):
        theta_old = omega.copy()

        # compute per-state reward
        reward = nominal_rewards - features @ omega

        # Backward, Forward
        policy = backward_causal(p_transition, reward, terminal, discount)
        d = forward(p_transition, p_initial, policy, terminal)

        # compute the gradient
        # df[i] is the expected visitation for feature i accross all (s, a, s_)
        # df[i] = [d[s, a, s_, i] * features[s, a, s_, i] for all (s, a, s_)].sum()
        df = (d[:, :, :, None] * features).sum((0, 1, 2))
        grad = df - e_features

        mean_error = np.abs(grad).mean()

        if epoch >= burnout and mean_error < best_error:
            best = omega.copy()
            best_error = mean_error

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        if omega.max() > max_penalty:
            omega = omega * (max_penalty / omega.max())
            optim.reset(omega)

        delta = np.max(np.abs(theta_old - omega))
        if epoch % 100 == 0:
            print(f'MAE(best): {min(mean_error, best_error): 0.15f}')
        epoch += 1

    print(f'Finished with MAE(best): {best_error: 0.15f}')

    return best if best is not None else omega


def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))
