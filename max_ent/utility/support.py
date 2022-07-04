from max_ent.gridworld import Directions
import max_ent.examples.grid_9_by_9 as G
import seaborn as sns
import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.spatial import distance
import random as r
import pickle
from scipy import stats

# allow us to re-use the framework from the src directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../')))

from mc.self import *
from mc.system1 import *
from mc.system2 import *
from mc.mca import *

def compute_mean(target = None, demo = None, constraints = None):

    df = pd.DataFrame()

    if target: 

        for i in range(0, len(target.time_stat)):
            #print(f"Stat: {mca.trajectory_stat[i]}")
            mask_1 = np.array(target.trajectory_stat[i]) == 1
            mask_2 = np.array(target.trajectory_stat[i]) == 0
            
            #Creaty np array from time array
            selected = np.array(target.time_stat[i])
            #Select actions in trajectory based on which system computed them
            selected_1= selected[mask_1]
            selected_2= selected[mask_2]
            #Compute total time per solver
            time_s1 = np.sum(selected_1)
            time_s2 = np.sum(selected_2)
            
            #Creaty np array from trajectory array
            selected = np.array(target.trajectory_stat[i])
            #Select builder in trajectory based on which system computed them
            selected_1= selected[mask_1]
            selected_2= selected[mask_2]
            #Compute total time per solver
            usage_s1 = np.sum(selected_1)
            usage_s2 = len(target.trajectory_stat[i]) - np.sum(selected_1)
            
            #Creaty np array from trajectory array
            selected = np.array(target.action_reward[i])
            #Select builder in trajectory based on which system computed them
            selected_1= selected[mask_1]
            selected_2= selected[mask_2]
            #Compute total time per solver
            reward_s1 = np.sum(selected_1)
            reward_s2 = np.sum(selected_2)
            
            
            violated_cs_1 = 0
            violated_cs_2 = 0
            violated_ca_1 = 0
            violated_ca_2 = 2
            if demo:
                selected = np.array(demo.trajectories[i].transitions())
                violated_cs_1= np.sum(np.isin(selected[mask_1][:,2], constraints['cs']))
                violated_cs_2= np.sum(np.isin(selected[mask_2][:,2], constraints['cs']))
                
                ca = [x.idx for x in constraints['ca']]
                violated_ca_1= np.sum(np.isin(selected[mask_1][:,1], ca))
                violated_ca_2= np.sum(np.isin(selected[mask_2][:,1], ca))
            

            dict_mca = {}
            dict_mca['traj_n'] = i
            dict_mca['length'] = len(target.trajectory_stat[i])
            dict_mca['reward'] = np.sum(target.action_reward[i])
            dict_mca['time'] = np.sum(target.time_stat[i])
            
            dict_mca['sub_type'] = "s1"
            dict_mca['time_agent'] = time_s1
            dict_mca['avg_time'] = time_s1 / usage_s1
            dict_mca['reward_agent'] = reward_s1
            dict_mca['avg_reward'] = reward_s1 / usage_s1
            dict_mca['usage']= usage_s1
            dict_mca['viol_constr']= violated_cs_1 + violated_ca_1
            dict_mca['perc_usage']= usage_s1 / len(target.trajectory_stat[i])
            temp_df = pd.DataFrame(data=dict_mca, index=[i])
            df = pd.concat([df, temp_df])
            
            
            dict_mca = {}
            dict_mca['traj_n'] = i
            dict_mca['length'] = len(target.trajectory_stat[i])
            dict_mca['reward'] = np.sum(target.action_reward[i])
            dict_mca['time'] = np.sum(target.time_stat[i])
            
            dict_mca['sub_type'] = "s2"
            dict_mca['time_agent'] = time_s2
            dict_mca['avg_time'] = time_s2 / usage_s2
            dict_mca['reward_agent'] = reward_s2
            dict_mca['avg_reward'] = reward_s2 / usage_s2
            dict_mca['usage']= usage_s2
            dict_mca['viol_constr']= violated_cs_2 + violated_ca_2
            dict_mca['perc_usage']= usage_s2 / len(target.trajectory_stat[i])
            
            temp_df = pd.DataFrame(data=dict_mca, index=[i])
            df = pd.concat([df, temp_df])
            
    else:
        dict_mca = {}
        dict_mca['traj_n'] = 0
        dict_mca['length'] = 0
        dict_mca['reward'] = 0
        dict_mca['time'] = 0

        dict_mca['sub_type'] = "null"
        dict_mca['time_agent'] = 0
        dict_mca['avg_time'] = 0 
        dict_mca['reward_agent'] = 0
        dict_mca['avg_reward'] = 0 
        dict_mca['usage']= 0
            
        temp_df = pd.DataFrame(data=dict_mca, index=[0])
        df = pd.concat([df, temp_df])

    
    #print(dict_mca)
    
    return df


def plot_results(df, x, y, min_label, max_label, bootstrap = 0):
    
    fig=plt.figure(figsize=(12, 7))
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)
    sns.color_palette("viridis", as_cmap=True)
    temp_df = df.loc[(df['type']!="S1 noMyopic")&(df['type']!="const")&(df['type']!="nominal")& (df['type']!="S1 Myopic") & (df['type']!="S2")& (df['traj_n']>=bootstrap)]
    #print("Prima")
    #g=sns.lineplot(x=x, y=y, data=df, hue="type",markers=True, dashes=False)
    g = sns.barplot(x=x, y=y, hue="type", data=temp_df, palette="autumn", ci=95);
    #print("Dopo")
    #g.set_xticklabels([f"({(i)/10:0.1f}, {1 - (i)/10:0.1f})" for i in range(11)])
    
    constrained_line = np.mean(df.loc[(df['type']=="const")][y])
    nominal_line = np.mean(df.loc[(df['type']=="nominal")][y])
    #print(f"constrained_line: {constrained_line} {y}")
    #print(f"nominal_line: {nominal_line} {y}")
    #s1_line = np.mean(df.loc[(df['type']=="s1")& (df['traj_n']>=bootstrap)][y])
    s2_line = np.mean(df.loc[(df['type']=="S2")& (df['traj_n']>=0)][y])
    #mixed_line = np.median(temp_df.loc[(temp_df['type']=="mixed")& (temp_df['traj_n']>=bootstrap)][y])
    s1nb_line_noMyopic = np.mean(df.loc[(df['type']=="S1 noMyopic")& (df['traj_n']>=0)][y])
    s1nb_line_Myopic = np.mean(df.loc[(df['type']=="S1 Myopic")& (df['traj_n']>=0)][y])
    
    #print(f"s2: {s2_line} s1:{s1nb_line}")
    
    g.axhline(constrained_line, color='r', linestyle='--', label="RL Constrained")
    g.axhline(nominal_line, color='b', linestyle='--', label="RL Nominal")
    #g.axhline(s1_line, color='b', linestyle='--', label="S1")
    #g.axhline(mixed_line, color='g', linestyle='--', label="Mixed")
    g.axhline(s1nb_line_noMyopic, color='b', linestyle='-.', label="S1 no Myopic")
    g.axhline(s1nb_line_Myopic, color='black', linestyle='-.', label="S1 Myopic")
    g.axhline(s2_line, color='g', linestyle='-.', label="S2")
    #g.set_ylim([min_label, max_label])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title(label=y+" varying "+x)
    
    '''h, l = g.get_legend_handles_labels()
    labels=["S1", "S2","SOFAI 10","SOFAI 01","SOFAI 02"]
    g.legend(h, labels)'''
    plt.legend()
    plt.grid(alpha=0.3)
    #g.set_xticks(range(11)) # <--- set the ticks first

    #plt.xlabel("W(Nominal, Constraints)")
    #plt.ylabel("Avg JS dist")
    plt.show()
    #fig.savefig(os.path.join("./", f"{y}_varying_{x}.png"), bbox_inches = 'tight')
    fig.savefig(os.path.join("./", f"{y}.pdf"), bbox_inches = 'tight')


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
    
    print(n.world.p_transition[0,0,:])

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


def count_states(trajectories, grid, nominal, constraints, bootstrap = 0, normalize = True, avoid_impossible = False):
    #grid = world.mdp
    count_matrix = np.ones((9, 9, 8, 9, 9)) * 1e-10
    #count_matrix = np.zeros((9, 9, 8, 9, 9))
    impossible_moves = 0
    total_transitions = 0
    if not normalize: count_matrix = np.zeros((9, 9, 8, 9, 9))
    avg_length = 0.0
    avg_reward = 0.0
    avg_reward_n = 0.0
    avg_violated = 0.0
    avg_cs = 0.0
    avg_ca = 0.0
    avg_cb = 0.0
    avg_cg = 0.0
    n = len(trajectories) -  bootstrap
    i=0
    for trajectory in trajectories:
        i += 1
        if i > bootstrap:
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
                total_transitions += 1
                # print(state)
                state_s = transition[0]
                action = transition[1]
                state_t = transition[2]
                increment = 1
                if (avoid_impossible and state_s == state_t): 
                    increment = 0
                    impossible_moves += 1
                count_matrix[grid.world.state_index_to_point(
                    state_s)][action][grid.world.state_index_to_point(state_t)] += increment

    if avoid_impossible:
        print(f"Imp. moves/Tot. Trans.: {impossible_moves}/ {total_transitions}")
    if normalize:
        return count_matrix / np.sum(count_matrix), avg_length / n, avg_reward / n, avg_reward_n / n, avg_violated/n, (avg_cs/n, avg_ca/n, avg_cb/n, avg_cg/n)
    else:
        return count_matrix, avg_length / n, avg_reward / n, avg_reward_n / n, avg_violated/n, (avg_cs/n, avg_ca/n, avg_cb/n, avg_cg/n)


# check for distance between start and terminal states
def generate_constraints(size, n_constraints=None):

    # generate the list of non-constrained states
    list_available = [x for x in range(size ** 2)]

    blue=[]
    if not n_constraints: blue = np.random.choice(list_available, 6, replace = False)  # blue states
    # remove blue states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, blue)

    green=[]
    if not n_constraints: green = np.random.choice(list_available, 6, replace = False)  # green states
    # remove green states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, green)

    cs = np.random.choice(list_available, n_constraints, replace = False)  # constrained states
    # remove constrained states from the list of non-constrained states
    list_available = np.setdiff1d(list_available, cs)

    # print(blue)
    # print(green)
    # print(cs)
    # print(list_available)

    random_ca = np.random.choice(8, 2, replace = False)  # green states
    ca = [Directions.ALL_DIRECTIONS[d] for d in random_ca]
    # print(ca)

    generate = True
    while generate:
        # generate start state from the list of non-constrained states
        start = np.random.choice(list_available)
        # generate terminal state from the list of non-constrained states
        goal = np.random.choice(list_available)

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

def build_dict(temp_matrix = None, type_mca = None, agent=None, s1_usage=0,  t1=200, t2=0.8, t3=0, t4=0, t6=1, t7=0.5, bootstrap=0, demo =None, constraints = None):
    '''temp_dict={}
    temp_dict['type']= type_mca
    temp_dict['Length']= temp_matrix[1]
    temp_dict['Reward']= temp_matrix[2]
    temp_dict['Viol'] = temp_matrix[4]
    temp_dict['S1_Usage'] = s1_usage
    temp_dict['t1'] = t1
    temp_dict['t2'] = t2
    temp_dict['t3'] = t3
    temp_dict['t4'] = t4
    temp_dict['t6'] = t6
    temp_dict['t7'] = t7
    temp_df = pd.DataFrame(data=temp_dict, index=[0])'''
    
    temp_df = compute_mean(agent, demo, constraints)
    temp_df['type']= type_mca
    
    if temp_matrix:
        temp_df['Length']= temp_matrix[1]
        if agent == None: temp_df['length']= temp_matrix[1]
        temp_df['Reward']= temp_matrix[2]
        if agent == None: temp_df['reward']= temp_matrix[2]
        temp_df['Viol'] = temp_matrix[4]
    else:
        temp_df['Length']= np.mean(temp_df['length'])
        temp_df['Reward']= np.mean(temp_df['reward'])
        temp_df['Viol'] = np.mean(temp_df['viol_constr'])
        
    temp_df['S1_Usage'] = s1_usage
    temp_df['t1'] = t1
    temp_df['t2'] = t2
    temp_df['t3'] = t3
    temp_df['t4'] = t4
    temp_df['t6'] = t6
    temp_df['t7'] = t7
    
    return temp_df

def simulation(n_cfg, c_cfg, demo, demo_mca_s1, demo_mca_s2, mca_s1, mca_s2, constraints, n_trajectories=200, 
               threshold1 = 200, threshold2 = 0.8, threshold3 = 0.4, 
               threshold4 = 200, threshold6 = 1, threshold7 = 0.5, df=None, jsdiv =None, bootstrap = 0):
    
    if df is None:
        df = pd.DataFrame()
        
    if jsdiv is None:
        jsdiv = pd.DataFrame()
   
    n=n_cfg.mdp
    c=c_cfg.mdp
    
    '''temp_matrix = count_states(demo.trajectories, c_cfg.mdp, n, constraints)
    temp_dict=build_dict(temp_matrix, type_mca='const')
    df = pd.concat([df, temp_dict])

    #mca_s1 = MCA(n=n, c=c, demo=demo, only_s1=True)
    #demo_mca_s1 = mca_s1.generate_trajectories(n_trajectories)
    temp_matrix_mca_s1 = count_states(demo_mca_s1.trajectories, c_cfg.mdp, n, constraints, bootstrap = bootstrap)
    temp_dict=build_dict(temp_matrix_mca_s1, type_mca='s1', agent=mca_s1, s1_usage=mca_s1.getStatistics()[0])
    df = pd.concat([df, temp_dict])

    #mca_s2 = MCA(n=n, c=c, demo=demo, only_s2=True)
    #demo_mca_s2 = mca_s2.generate_trajectories(n_trajectories)
    temp_matrix_mca_s2 = count_states(demo_mca_s2.trajectories, c_cfg.mdp, n, constraints, bootstrap = bootstrap)
    temp_dict=build_dict(temp_matrix_mca_s2, type_mca='s2', agent=mca_s2 )
    df = pd.concat([df, temp_dict])'''
    
    #for t3 in threshold3:
    #    for t4 in threshold4:
    #print(f"t1:{t1} t2:{t2} t3:{t3} t4:{t4} t6:{t6} t7:{t7} ")

    mca_10 = MCA(n=n, c=c, demo=None, threshold1 = threshold1,  threshold3 = threshold3, threshold4 = threshold4, threshold5 = 0)
    demo_mca_10 = mca_10.generate_trajectories(n_trajectories)
    temp_matrix_mca_10 = count_states(demo_mca_10.trajectories, c_cfg.mdp, n, constraints, bootstrap = bootstrap)
    temp_dict=build_dict(temp_matrix_mca_10, type_mca='10', agent= mca_10, s1_usage=mca_10.getStatistics()[0], t1=threshold1, t3=threshold3, t4=threshold4)
    f1 = G.plot_world(f'MCA 10', c, c_cfg.state_penalties, c_cfg.action_penalties, c_cfg.color_penalties, demo_mca_10, c_cfg.blue, c_cfg.green, vmin=-50, vmax=10)
    df = pd.concat([df, temp_dict])


    mca_01 = MCA(n=n, c=c, demo=None, threshold1 = threshold1, threshold3 = threshold3, threshold4 = threshold4, threshold5 = 1)
    demo_mca_01 = mca_01.generate_trajectories(n_trajectories)
    temp_matrix_mca_01 = count_states(demo_mca_01.trajectories, c_cfg.mdp, n, constraints, bootstrap = bootstrap)
    temp_dict=build_dict(temp_matrix_mca_01, type_mca='01', agent=mca_01, s1_usage=mca_01.getStatistics()[0], t1=threshold1,t3=threshold3, t4=threshold4)
    f1 = G.plot_world(f'MCA 01', c, c_cfg.state_penalties, c_cfg.action_penalties, c_cfg.color_penalties, demo_mca_01, c_cfg.blue, c_cfg.green, vmin=-50, vmax=10)
    df = pd.concat([df, temp_dict])


    mca_02 = MCA(n=n, c=c, demo=None, threshold1 = threshold1, threshold3 = threshold3, threshold4 = threshold4, threshold5 = 2)
    demo_mca_02 = mca_02.generate_trajectories(n_trajectories)
    temp_matrix_mca_02 = count_states(demo_mca_02.trajectories, c_cfg.mdp, n, constraints, bootstrap = bootstrap)
    temp_dict=build_dict(temp_matrix_mca_02, type_mca='02', agent=mca_02, s1_usage=mca_02.getStatistics()[0],  t1=threshold1,t3=threshold3, t4=threshold4)
    df = pd.concat([df, temp_dict])
    #print(mca_02.__dict__)
    f1 = G.plot_world(f'MCA 02', c, c_cfg.state_penalties, c_cfg.action_penalties, c_cfg.color_penalties, demo_mca_02, c_cfg.blue, c_cfg.green, vmin=-50, vmax=10)

    
    temp_jsdiv = {}
    temp_jsdiv['t3'] = threshold3
    temp_jsdiv['t4'] = threshold4
    temp_jsdiv['jsdiv'] = js_divergence((mca_s2.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_s2.modelSelf.ntra_per_transition + 1E-10), (mca_01.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_01.modelSelf.ntra_per_transition + 1E-10))
    temp_jsdiv['type'] = '01'
    temp_jsdiv = pd.DataFrame(data=temp_jsdiv, index=[0])
    jsdiv = pd.concat([jsdiv, temp_jsdiv])
    
    temp_jsdiv = {}
    temp_jsdiv['t3'] = threshold3
    temp_jsdiv['t4'] = threshold4
    temp_jsdiv['jsdiv'] = js_divergence((mca_s2.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_s2.modelSelf.ntra_per_transition + 1E-10), (mca_10.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_10.modelSelf.ntra_per_transition + 1E-10))
    temp_jsdiv['type'] = '10'
    temp_jsdiv = pd.DataFrame(data=temp_jsdiv, index=[0])
    jsdiv = pd.concat([jsdiv, temp_jsdiv])
    
    temp_jsdiv = {}
    temp_jsdiv['t3'] = threshold3
    temp_jsdiv['t4'] = threshold4
    temp_jsdiv['jsdiv'] = js_divergence((mca_s2.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_s2.modelSelf.ntra_per_transition + 1E-10), (mca_02.modelSelf.ntra_per_transition + 1E-10)/np.sum(mca_02.modelSelf.ntra_per_transition + 1E-10))
    temp_jsdiv['type'] = '02'
    temp_jsdiv = pd.DataFrame(data=temp_jsdiv, index=[0])
    jsdiv = pd.concat([jsdiv, temp_jsdiv])
                            
    return df, jsdiv
