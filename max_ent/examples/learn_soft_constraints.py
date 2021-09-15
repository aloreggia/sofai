from collections import defaultdict
from max_ent.examples.compare_hard_results import main
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt
import numpy as np
from max_ent.gridworld import Directions
from max_ent.algorithms.gridworld_icrl import ICRL_Result, learn_constraints, convert_constraints_to_probs, generate_trajectories, MDP
import json
import pickle
from pathlib import Path

lens = list(range(1, 10)) + list(range(10, 101, 10))


def run_game(d, p_slip):
    np.random.seed(123)
    N = 10
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
    n, c = n_cfg.mdp, c_cfg.mdp

    results = [[] for _ in range(N)]
    for i in range(N):
        # Generate demonstrations for the i-th set
        demo = generate_trajectories(
            c.world, c.reward, c.start, c.terminal, n_trajectories=100)
        for L in lens:
            print(f'game #{d["game"]}, set #{i+1}, len={L}')
            # Learn the constraints
            r = learn_constraints(
                n.reward, c.world, c.terminal, demo.trajectories[:L])
            results[i].append({'reward': r.reward, 'omega': r.omega})

    return results


def run(data_file, out_path, deterministic):
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    p_slip = 0 if deterministic else 0.1
    with open(data_file) as f:
        data = json.load(f)
    results = []
    for d in data:
        results.append(run_game(d, p_slip))
        print(f'Saving checkpoint game {d["game"]}')
        with out_path.open('wb') as f:
            pickle.dump(results, f)


def main():
    print('Running deterministic games ...')
    run('data/random_data_10.json',
        'results/soft/random_data_10_deter.pkl', True)

    print('Running non deterministic games ...')
    run('data/random_data_10.json',
        'results/soft/random_data_10_non_deter.pkl', False)

    print('Running scobee example ...')
    run('data/scobee_example_data.json',
        'results/soft/scobee_example_data_deter.pkl', True)
    run('data/scobee_example_data.json',
        'results/soft/scobee_example_data_non_deter.pkl', False)

    print('Done!')


if __name__ == '__main__':
    main()
