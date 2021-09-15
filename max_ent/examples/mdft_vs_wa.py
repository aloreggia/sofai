from math import sqrt
from numpy.core.fromnumeric import size
from max_ent.utility.support import js_divergence
from mdft_nn import mdft, mdft_net, trainer, trainer_helpers
import numpy as np
from max_ent.algorithms import rl as RL


def main():
    np.random.seed(0)
    n = 100
    jsd = np.zeros(n)
    for i in range(n):
        print(i+1)
        q_n = np.random.randn(1, 8) * 10
        q_c = np.random.randn(1, 8) * 10
        p_n = RL.stochastic_policy_from_q_value(None, q_n)
        p_c = RL.stochastic_policy_from_q_value(None, q_c)
        M = np.concatenate([p_n.T, p_c.T], 1)
        w = np.random.rand() * 0.6 + 0.2
        w = np.array([[w], [1 - w]])
        dist = p_n * w[0] + p_c * w[1]
        idx = [list(range(M.shape[0]))]
        d = {
            'pref_based': False,
            'M': M.tolist(),
            'b': 2,
            'phi1': 0.01,
            'phi2': 0.01,
            'sig2': 1,
            'threshold': 25,
            'w': w.tolist(),
            'D': dist.tolist(),
            'idx': idx
        }
        opts = {
            'nprint': 30,
            'ntest': 10000,
            'ntrain': 100,
            'niter': 300,
            # 'w_decay': 1,
            # 'w_lr': 0.001,
            'grad_clip': 5,
            'm': False,
            'w': True,
            's': False
        }
        learned, it = trainer.train(d, opts)

        pred = trainer_helpers.get_model_dist(learned, d, 1000)
        pred = np.array(pred) + 1e-6
        pred /= pred.sum()

        dist += 1e-6
        dist /= dist.sum()
        jsd[i] = js_divergence(pred, dist)

    np.set_printoptions(precision=3, suppress=True)
    print(f'mean: {jsd.mean()}, se: {jsd.std()/sqrt(n)}')


if __name__ == "__main__":
    main()
