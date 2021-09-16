import numpy as np

class ModelSelf:
	"""
	The Model of Self store all the known trajectories. This is useful to compute statistics and compute new actions
	__grid: is the world on which we are making experience
	__n: number of trajectories experienced in the world
	__std: standard deviations
	__statistics: for each transition, it keeps track of how many trajectory use that transition
	"""
	def __init__(self, grid, demonstrations = []):
		self.statistics = np.ones((9, 9, 8, 9, 9)) * 1e-10
		self.n = 0
		self.std = 0
		self.grid = grid
		for t in demonstrations:
			self.updateModel(t)

	def getStart(self):
		return self.grid.start[0]

	def getGoal(self):
		return self.grid.terminal[0]

	def getNStates(self):
		return self.grid.world.n_states

	def getWorld(self):
		return self.grid.world

	def updateModel(self, trajectory):
		"""
		Given a new trajectory, update the model of self.
		"""
		self.n +=1
		for transition in trajectory.transitions():
			# print(state)
			state_s = transition[0]
			action = transition[1]
			state_t = transition[2]
			self.statistics[self.grid.world.state_index_to_point(state_s)][action][self.grid.world.state_index_to_point(state_t)] += 1

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
		#print(f"state: {state_s} \t action: {action}")
		move_cell_trajectories = np.sum(self.statistics[self.grid.world.state_index_to_point(state_s)][action])
		trajectories_state = self.getNTrajectories(state_s)

		reward = move_cell_trajectories / trajectories_state
		confidence = reward / self.std

		return reward, confidence

	def getNTrajectories(self, state_s):
		return np.sum(self.statistics[self.grid.world.state_index_to_point(state_s)])