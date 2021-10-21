import numpy as np
from scipy.special import expit

class ModelSelf:
	"""
	The Model of Self store all the known trajectories. This is useful to compute ntra_per_transition and compute new actions
	__grid: is the world on which we are making experience
	__n: number of trajectories experienced in the world
	__std: standard deviations
	__statistics: for each transition, it keeps track of how many trajectory use that transition
	__part_reward: for each state, it keeps track of the cumulative partial reward to reach the state
	__prob: for each pair (state, action), it keeps track of the probability of taking the action in the state
	"""
	def __init__(self, grid, constrained, demo):
		self.ntra_per_transition = np.zeros((9, 9, 8, 9, 9)) #* 1e-10
		self.ntra_per_transition_s2 = np.zeros((9, 9, 8, 9, 9)) #* 1e-10
		self.ntra_per_stateAction = np.zeros((9, 9, 8))
		self.ntra_per_stateAction_s2 = np.zeros((9, 9, 8))
		self.ntra_per_state = np.zeros((9, 9))
		self.ntra_per_state_s2 = np.zeros((9, 9))
		self.part_reward = np.zeros((9, 9)) 
		self.part_reward_s2 = np.zeros((9, 9)) 
		self.part_reward_state_action = np.zeros((9, 9, 8)) 
		self.prob = np.ndarray(shape=(9,9,8), dtype=object)
		self.n = 0
		self.std = 0
		self.grid = grid
		self.constraints = constrained

		if demo != None:
			for t in demo.trajectories:
				self.updateModel(t)

	def getStart(self):
		return self.grid.start[0]

	def getGoal(self):
		return self.grid.terminal[0]

	def getNStates(self):
		return self.grid.world.n_states

	def getWorld(self):
		return self.grid.world

	def getTotalReward(self,trajectory):
		reward = 0.0
		for state in trajectory.transitions():
			reward += self.constraints.reward[state]
		
		return reward

	def updateModel(self, trajectory, traj_builders=None):
		"""
		Given a new trajectory, update the model of self.
		"""
		self.n +=1
		reward = self.getTotalReward(trajectory)
		#print(self.part_reward)
		temp_ntra_per_transition = np.zeros((9, 9, 8, 9, 9))
		temp_ntra_per_stateAction = np.zeros((9, 9, 8))
		temp_ntra_per_state = np.zeros((9, 9))
		temp_prob = np.ndarray(shape=(9,9,8), dtype=object)
		i = 0
		for transition in trajectory.transitions():
			# print(state)
			state_s = transition[0]
			action = transition[1]
			state_t = transition[2]

			#compute the coordinates for initial and final state
			state_s_coord = self.grid.world.state_index_to_point(state_s)
			state_t_coord = self.grid.world.state_index_to_point(state_t)

			#update the number of traj for the transition
			if temp_ntra_per_transition[state_s_coord][action][state_t_coord] == 0:
				temp_ntra_per_transition[state_s_coord][action][state_t_coord] += 1

			#update the number of traj for the (state, action)
			if temp_ntra_per_stateAction[state_s_coord][action] == 0:
				temp_ntra_per_stateAction[state_s_coord][action] += 1

			#update the number of traj for the state
			if temp_ntra_per_state[state_s_coord] == 0:
				temp_ntra_per_state[state_s_coord] += 1

			#Update the partial reward for a given state
			#if not temp_part_reward[state_t_coord]:
			self.part_reward[state_t_coord] += self.constraints.reward[transition]
			self.part_reward_state_action[state_t_coord][action] += self.constraints.reward[transition]

			#create the dictionary for probabilities
			if not self.prob[state_s_coord][action]: 
				self.prob[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			#create the dictionary for probabilities
			if not temp_prob[state_s_coord][action]: 
				temp_prob[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			if reward not in temp_prob[state_s_coord][action].keys(): 
				temp_prob[state_s_coord][action][reward] = 1

			i += 1
			#self.prob[state_s_coord][action]['tot_trj'] += 1

		#update the number of traj for the terminal state
		if temp_ntra_per_state[state_t_coord] == 0:
			temp_ntra_per_state[state_t_coord] += 1

		#populate le dictionary, each reward is a key and the value is the number of trajectory 
		# passing through state_s, action with that reward
		for x in range(temp_prob.shape[0]):
			for y in range(temp_prob.shape[1]):	
				for d in range(temp_prob.shape[2]):
					if temp_prob[x][y][d]:
						for key in temp_prob[x][y][d].keys():
							if key not in self.prob[x,y][d].keys(): 
								self.prob[x,y][d][key] = temp_prob[x][y][d][key]
							else: 
								self.prob[x,y][d][key] += temp_prob[x][y][d][key]

		self.ntra_per_transition  = self.ntra_per_transition  + temp_ntra_per_transition 
		self.ntra_per_stateAction  = self.ntra_per_stateAction  + temp_ntra_per_stateAction 
		self.ntra_per_state  = self.ntra_per_state  + temp_ntra_per_state 
		self.std = np.std(self.ntra_per_transition  / np.sum(self.ntra_per_transition))

	def updateIfZero(number):
		if number==0: 
			return 1
		return 0

	def getReward(self, state_s, action):
		"""
		Given:
		states: the actual position in the world
		action: an action to be taken

		Compute: 
		reward as the probability that a trajectory from state 'state_s' takes action 'action' 
		confidence as the ratio between the reward and the normalized standard deviation
		"""
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		#if (state,action) never appeared before then return -inf with the highest confidence
		if not self.prob[state_s_coord][action]:
			return float('-inf'), 1

		temp_list = []
		#print(f"{self.prob[state_s_coord][action]}")
		'''
		build the distribution of reward * prob where:
			key is the reward and the dictionary value associated to the key is the number of traj
			through state action with that reward
		'''
		temp_ntra_per_stateAction = np.sum(self.ntra_per_stateAction[state_s_coord])
		for key in self.prob[state_s_coord][action].keys():
			if key is not 'tot_trj':
				temp_reward = key
				temp_prob = self.prob[state_s_coord][action][key] / self.ntra_per_stateAction[state_s_coord][action]
				temp_list.append(temp_reward * temp_prob)
				#temp_list.append(self.prob[state_s_coord][action][key]*(key/self.prob[state_s_coord][action]['tot_trj']))

		exp_reward = np.sum(temp_list)
		temp_list = temp_list/exp_reward
		#print(f"temp_list_norm: {temp_list/exp_reward}")
		#print(f"state: {state_s} \t action: {action} \t {self.prob[state_s_coord][action]}")
		#print(f"templist: {temp_list}\n")
		#move_cell_trajectories = self.prob[state_s_coord][action]['tot_trj']
		#print(f"move: {move_cell_trajectories}\t tot_trj {self.prob[state_s_coord][action]['tot_trj']}")
		#trajectories_per_state = self.getNTrajectories(state_s)

		#r = move_cell_trajectories / trajectories_per_state
		#print(f"r_pre: {r}")
		r = self.ntra_per_stateAction[state_s_coord][action] / temp_ntra_per_stateAction
		#print(f"r_post: {r}")
		temp_std = np.std(temp_list)
		confidence = expit(r / (temp_std + 1e-10))
		#print(f"exp_rew: {exp_reward} \t temp_std: {temp_std} \t r: {r} \t confidence: {confidence}")

		return exp_reward, confidence

	def getNTrajectories(self, state_s):
		#return np.sum(self.ntra_per_transition[self.grid.world.state_index_to_point(state_s)])
		return self.ntra_per_state[self.grid.world.state_index_to_point(state_s)]

	def getNTrajectoriesStateAction(self, state_s, action):
		#return np.sum(self.ntra_per_transition[self.grid.world.state_index_to_point(state_s)])
		return self.ntra_per_stateAction[self.grid.world.state_index_to_point(state_s)][action]

	def getAvgPartialReward(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		#print(f"part_reward: {self.part_reward[state_s_coord]} \t tot_traj: {np.sum(self.ntra_per_transition[state_s_coord])}")
		return self.part_reward[state_s_coord] / self.getNTrajectories(state_s)

	def updatePartRewardS2(self, transition):
		state_s_coord = self.grid.world.state_index_to_point(transition[0])
		self.part_reward_s2[state_t_coord] += self.constraints.reward[transition]
		self.ntra_per_state_s2[state_t_coord] += 1

	def getAvgPartialRewardStateAction(self, state_s, action):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		return self.part_reward_state_action[state_s_coord][action] / self.getNTrajectoriesStateAction(state_s, action)

	def getMinMaxPartialReward(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		min_rew = float('+inf')
		max_rew = float('-inf')
		for action in self.prob[state_s_coord]:
			if action:
				min_rew = min(min_rew,min(action.keys()))
				max_rew = max(max_rew,max(action.keys()))

		return min_rew, max_rew