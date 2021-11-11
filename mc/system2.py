import numpy as np
from itertools import chain
from mdft_nn.mdft import MDFT, get_time_based_dft_dist, get_preference_based_dft_dist
from mdft_nn.helpers.distances import hotaling_S
from max_ent.algorithms import rl as RL


class System2Solver:
	def __init__(self):
		
		super()

	def policy(self, modelSelf, state, w=[1,0]):

		#per ogni azione computa la probabilità che appartenga a una traiettoria positiva
		#argmax prob dell'azione 

		# set up initial probabilities for trajectory generation
		initial = np.zeros(modelSelf.getNStates())
		initial[modelSelf.getStart()] = 1.0

		discount = 0.9
		normalize = True

		# generate trajectories
		q_n, _ = RL.value_iteration(modelSelf.getWorld().p_transition, modelSelf.grid.reward, discount)
		q_c, _ = RL.value_iteration(modelSelf.getWorld().p_transition, modelSelf.constraints.reward, discount)
		if normalize:
			policy_n = RL.stochastic_policy_from_q_value(modelSelf.getWorld(), q_n)
			policy_c = RL.stochastic_policy_from_q_value(modelSelf.getWorld(), q_c)

		policy_exec = mdft_policy_adapter(policy_n, policy_c, w=np.array(w))

		action = policy_exec(state)

		return action

def mdft_policy_adapter(nominal_q, constrained_q, w=None, pref_t = 10):
	def policy(state):
		r = nominal_q[state]
		c = constrained_q[state]
		M = np.concatenate([r[:, None], c[:, None]], 1)
		S = hotaling_S(M, 0.01, 0.01, 2)
		p0 = np.zeros((M.shape[0], 1))
		mdft = MDFT(M, S, w, p0)
		#dist = get_time_based_dft_dist(mdft, 1, delib_t)
		dist = get_preference_based_dft_dist(mdft, 1, pref_t)
		return np.argmax(dist)

	return policy