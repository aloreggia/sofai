from max_ent.gridworld.gridworld import Directions
from max_ent.examples.grid_9_by_9 import config_world, generate_random_config, generate_trajectories
import numpy as np
import json
from pathlib import Path


def generate_random():
    N = 10
    games = []
    i = 0
    while i < N:
        n_cfg, c_cfg = generate_random_config(
            min_dist_start_goal=8, p_slip=0, penalty=-50, dist_penalty=True)
        trj = generate_trajectories(c_cfg.mdp.world, c_cfg.mdp.reward, c_cfg.mdp.start,
                                    c_cfg.mdp.terminal, n_trajectories=10).trajectories
        path_to_goal = 0
        # Check if goal is achievable
        for t in trj:
            violation = False
            last = None
            for s, a, s_ in t.transitions():
                last = s
                if c_cfg.mdp.reward[s, a, s_] < -10:  # Violation
                    violation = True
                    break

            path_to_goal += 0 if violation or last != c_cfg.mdp.terminal[0] else 1

            if path_to_goal >= 3:  # Found enough paths to the goal
                break

        if path_to_goal < 3:
            print('.', end='')
            continue

        games.append({
            'game': i + 1,
            'start': int(c_cfg.mdp.start[0]),
            'goal': int(c_cfg.mdp.terminal[0]),
            'blue': c_cfg.blue.tolist(),
            'green': c_cfg.green.tolist(),
            'state_reward': c_cfg.state_penalties.tolist(),
            'action_reward': [c_cfg.action_penalties[a] for a in Directions.ALL_DIRECTIONS],
            'color_reward': c_cfg.color_penalties.tolist(),
        })
        print(i+1)
        i += 1
    out = Path(f"./data/random_data_{N}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(games, f, indent=4)


def scobee_example():
    goal = 8
    blue = [4, 13, 22]  # blue states
    green = [58, 67, 76]  # green states
    cs = [31, 39, 41, 47, 51]  # constrained states
    ca = [Directions.UP_LEFT, Directions.UP_RIGHT]  # constrained actions
    cc = [1, 2]  # constrained colors

    c_cfg = config_world(blue, green, cs, ca, cc, goal)
    game = {
        'game': 1,
        'start': int(c_cfg.mdp.start[0]),
        'goal': int(c_cfg.mdp.terminal[0]),
        'blue': c_cfg.blue,
        'green': c_cfg.green,
        'state_reward': c_cfg.state_penalties.tolist(),
        'action_reward': [c_cfg.action_penalties[a] for a in Directions.ALL_DIRECTIONS],
        'color_reward': c_cfg.color_penalties.tolist(),
    }

    out = Path(f"./data/scobee_example_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump([game], f, indent=4)


if __name__ == '__main__':
    scobee_example()
    np.random.seed(123)
    generate_random()
