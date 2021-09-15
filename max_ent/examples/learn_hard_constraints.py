import json
from pathlib import Path
from max_ent.gridworld.gridworld import Directions
from max_ent.gridworld.trajectory import Trajectory
from max_ent.algorithms.gridworld_icrl import Demonstration, MDP, convert_constraints_to_probs, generate_hard_trajectories, generate_trajectories, learn_constraints
import numpy as np
from max_ent.examples.grid_9_by_9 import config_world, plot_world
import matplotlib.pyplot as plt

lens = list(range(1, 10)) + list(range(10, 101, 10))

def main():
    np.random.seed(123)
    thresholds = [0.4, 0.5, 0.6, 0.7]
    with open('data/scobee_example_data.json') as f:
        data = json.load(f)

    with open('data/scobee_example_trajectories.json') as f:
        games = json.load(f)
        g_trajs = {g['game']: [list(
            zip(t['state_seq'], t['action_seq'])) for t in g['trajs']] for g in games}

    for d in data:
        learned = learn_game(d, g_trajs, thresholds)

        for t in thresholds:
            cons = []
            soft_demos = []
            hard_demos = []
            for i in range(10):
                soft_demos.append([])
                hard_demos.append([])
                cons.append([])
                for l in lens:
                    cons[i].append(learned[i, l, t]['constraints'])
                    soft_demos[i].append(learned[i, l, t]['soft_demo'])
                    hard_demos[i].append(learned[i, l, t]['hard_demo'])
                    
            path = Path(f'results/hard/our_results_scobee_example_t{t}.json')
            path.parent.mkdir(exist_ok=True, parents=True)
            with path.open('w') as f:
                json.dump([{
                    'id': d['game'],
                    'learned_constraints': cons,
                    'soft_demos': soft_demos,
                    'hard_demos': hard_demos
                }], f, indent=4)


def learn_game(d, g_trajs, thresholds):
    cs = np.argwhere(np.array(d['state_reward']) == -50).ravel()
    ca = np.argwhere(np.array(d['action_reward']) == -50).ravel()
    ca = [Directions.ALL_DIRECTIONS[d] for d in ca]
    cc = np.argwhere(np.array(d['color_reward']) == -50).ravel()

    n_cfg = config_world(d['blue'], d['green'], [],
                         [], [], d['goal'], start=d['start'], p_slip=0)
    c_cfg = config_world(d['blue'], d['green'], cs,
                         ca, cc, d['goal'], start=d['start'], p_slip=0)
    n, c = n_cfg.mdp, c_cfg.mdp
    
    learned_constraints = {}
    for i, trajs in enumerate(g_trajs[d['game']]):
        for l in lens:
            print(f"Learning game #{d['game']}, set #{i+1}, len={l}")

            # Generate demonstrations and plot the world
            trj = [list(zip(s, a)) for s, a in trajs[:l]]
            trj = [Trajectory(
                [(s, a, c_cfg.mdp.world.state_index_transition(s, a)) for s, a in t]) for t in trj]
            demo = Demonstration(trj, None)

            # Learn the constraints
            result = learn_constraints(
                n.reward, c.world, c.terminal, demo.trajectories)
            result.state_weights[c_cfg.mdp.start] = 0
            learned = MDP(c.world, result.reward, c.terminal, c.start)

            p_s, p_a, p_c = convert_constraints_to_probs(n.reward, result)

            p_a_v = np.array([p_a[a] for a in Directions.ALL_DIRECTIONS])
            soft_demo = generate_trajectories(
                learned.world, learned.reward, learned.start, learned.terminal, n_trajectories=100)

            for threshold in thresholds:
                cs = [{"type": "state", "value": s.tolist()[0], "p": p.tolist()}
                      for s, p in zip(np.argwhere(p_s > threshold), p_s[p_s > threshold])]

                ca = [{"type": "action", "value": a.tolist()[0], "p": p.tolist()}
                      for a, p in zip(np.argwhere(p_a_v > threshold), p_a_v[p_a_v > threshold])]

                cc = [{"type": "feature", "value": c.tolist()[0], "p": p.tolist()}
                      for c, p in zip(np.argwhere(p_c > threshold), p_c[p_c > threshold])]

                cons = cs + ca + cc

                cons = sorted(cons, key=lambda x: -x['p'])

                hard_demo = get_hard_demo(d, cs, cc, ca, threshold)

                learned_constraints[i, l, threshold] = {
                    'constraints': cons,
                    'soft_demo': [serlialize_demo(soft_demo)],
                    'hard_demo': [serlialize_demo(hard_demo)]
                }

    return learned_constraints


def serlialize_demo(demo):
    return {
        'state_seq': [[int(s) for s, _, _ in t.transitions()] for t in demo.trajectories],
        'action_seq': [[int(a) for _, a, _ in t.transitions()] for t in demo.trajectories]
    }


def get_hard_demo(d, cs, cc, ca, threshold):
    cs = [c['value'] for c in cs if c['p'] > threshold]
    ca = [Directions.ALL_DIRECTIONS[c['value']]
          for c in ca if c['p'] > threshold]
    cc = [c['value'] for c in cc if c['p'] > threshold]

    h_cfg = config_world(d['blue'], d['green'], cs,
                         ca, cc, d['goal'], start=d['start'], p_slip=0, penalty=-200)
    h = h_cfg.mdp
    s_cons = cs.copy()
    a_cons = [a.idx for a in ca]
    if 1 in cc:
        s_cons += d['blue']
    if 2 in cc:
        s_cons += d['green']
        
    hard_demo = generate_hard_trajectories(
        h.world, h.reward, h.start, h.terminal, s_cons, a_cons, n_trajectories= 100)

    return hard_demo


if __name__ == '__main__':
    main()
    input("Press any key ...")
