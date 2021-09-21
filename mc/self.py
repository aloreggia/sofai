import numpy as np

class ModelSelf:
	"""
	The Model of Self store all the known trajectories. This is useful to compute statistics and compute new actions
	__grid: is the world on which we are making experience
	__n: number of trajectories experienced in the world
	__std: standard deviations
	__statistics: for each transition, it keeps track of how many trajectory use that transition
	"""
	def __init__(self, grid, constrained, demo):
		self.statistics = np.ones((9, 9, 8, 9, 9)) * 1e-10
		self.prob = np.ndarray(shape=(9,9,8), dtype=object)
		self.n = 0
		self.std = 0
		self.grid = grid
		self.constraints = constrained

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

	def updateModel(self, trajectory):
		"""
		Given a new trajectory, update the model of self.
		"""
		self.n +=1
		reward = self.getTotalReward(trajectory)
		for transition in trajectory.transitions():
			# print(state)
			state_s = transition[0]
			action = transition[1]
			state_t = transition[2]

			#compute the coordinates for initial and final state
			state_s_coord = self.grid.world.state_index_to_point(state_s)
			state_t_coord = self.grid.world.state_index_to_point(state_t)

			#update the number of traj for the transition
			self.statistics[state_s_coord][action][state_t_coord] += 1

			#create the dictionary for probabilities
			if not self.prob[state_s_coord][action]: 
				self.prob[state_s_coord][action]={}
				self.prob[state_s_coord][action]['tot_trj'] = 0
			
			#populate le dictionary, each reward is a key and the value is the number of trajectory 
			# passing through state_s, action with that reward
			#print(self.prob[state_s_coord][action].keys())
			#print(reward)
			if reward not in self.prob[state_s_coord][action].keys(): 
				self.prob[state_s_coord][action][reward] = 1
			else: 
				self.prob[state_s_coord][action][reward] += 1

			self.prob[state_s_coord][action]['tot_trj'] += 1

		self.std = np.std(self.statistics / np.sum(self.statistics))

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
		if not self.prob[state_s_coord][action]:
			return float('-inf'), 1

		temp_list = []
		#print(f"{self.prob[state_s_coord][action]}")
		#build the list distribution of rewards
		for key in self.prob[state_s_coord][action].keys():
			if key is not 'tot_trj':
				temp_list.append(self.prob[state_s_coord][action][key]*(key/self.prob[state_s_coord][action]['tot_trj']))

		exp_reward = np.sum(temp_list)
		#print(f"state: {state_s} \t action: {action} \t {self.prob[state_s_coord][action]}")
		#print(f"templist: {temp_list}\n")
		move_cell_trajectories = np.sum(self.statistics[state_s_coord][action])
		trajectories_per_state = self.getNTrajectories(state_s)

		r = move_cell_trajectories / trajectories_per_state
		confidence = r / np.std(temp_list)

		return exp_reward, confidence

	def getNTrajectories(self, state_s):
		return np.sum(self.statistics[self.grid.world.state_index_to_point(state_s)])