from functools import update_wrapper
from tkinter import TRUE
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
	def __init__(self, grid, constrained, demo, risk_aversion=1):
		self.ntra_per_transition = np.zeros((9, 9, 8, 9, 9)) #* 1e-10
		self.ntra_per_transition_s2 = np.zeros((9, 9, 8, 9, 9)) #* 1e-10
		self.ntra_per_stateAction = np.zeros((9, 9, 8))
		self.ntra_per_stateAction_s2 = np.zeros((9, 9, 8))
		self.ntra_per_state = np.zeros((9, 9))
		self.ntra_per_state_s2 = np.zeros((9, 9))
		self.part_length = np.ndarray(shape=(9,9), dtype=object)
		self.part_reward = np.zeros((9, 9)) 
		self.part_reward_s2 = np.zeros((9, 9)) 
		self.part_reward_state_action = np.zeros((9, 9, 8)) 
		self.prob = np.ndarray(shape=(9,9,8), dtype=object)
		self.prob_remaining = np.ndarray(shape=(9,9,8), dtype=object)
		self.prob_s2 = np.ndarray(shape=(9,9,8), dtype=object)
		self.prob_time = np.ndarray(shape=(9,9,8), dtype=object)
		self.n = 0
		self.std = 0
		self.grid = grid
		self.constraints = constrained
		self.avg_reward = 0
		self.total_transitions_s1 = 0
		self.total_transitions_s2 = 0
		self.s1_wrong = 0
		self.s1_wrong_list = []

		self.s2_wrong = 0
		self.s2_wrong_list = []

		self.risk_aversion = risk_aversion

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

	def getPartialRemainingReward(self, trajectory, index_start, time_stat):
		'''reward = 0.0
		reward = float("-inf")
		find = False

		total_time = float("+inf")

		for index, state in enumerate(trajectory.transitions()):
			if (state == transition):
				if not find: 
					reward = 0.0
					total_time = 0.0
				find=True

			if find:
				if self.constraints.reward[state] < -6:
					reward += -abs(self.constraints.reward[state] ** (1+self.risk_aversion))
				else:
					reward += self.constraints.reward[state]

				total_time += time_stat[index]'''

		reward = 0.0
		total_time = 0.0

		

		for index in range(index_start, len(trajectory.transitions())):
			state = trajectory.transitions()[index]
		
			if self.constraints.reward[state] < -6:
				reward += -abs(self.constraints.reward[state] ** (1+self.risk_aversion))
			else:
				reward += self.constraints.reward[state]

			total_time += time_stat[index]
		
		total_time = total_time / (len(trajectory.transitions()) -  (index_start))
		#print(f"index_start: {index_start} \t total_time: {total_time} \t len: {len(trajectory.transitions())} \n time: {time_stat} \n\n")

		return reward, total_time 

	def updateModel(self, trajectory, traj_builders=None, time_stat = None):
		"""
		Given a new trajectory, update the model of self.
		"""
		self.n +=1
		total_reward = self.getTotalReward(trajectory)
		#print(self.part_reward)
		temp_ntra_per_transition = np.zeros((9, 9, 8, 9, 9))
		temp_ntra_per_stateAction = np.zeros((9, 9, 8))
		temp_ntra_per_state = np.zeros((9, 9))
		temp_prob = np.ndarray(shape=(9,9,8), dtype=object)

		temp_ntra_per_transition_s2 = np.zeros((9, 9, 8, 9, 9))
		temp_ntra_per_stateAction_s2 = np.zeros((9, 9, 8))
		temp_ntra_per_state_s2 = np.zeros((9, 9))
		temp_prob_s2 = np.ndarray(shape=(9,9,8), dtype=object)

		temp_prob_time = np.ndarray(shape=(9,9,8), dtype=object)

		#self.total_transitions += len(trajectory.transitions())

		i = 0
		temp_partial_reward = 0
		for index, transition in enumerate(trajectory.transitions()):

			self.avg_reward += self.constraints.reward[transition]
			#self.total_transitions += 1

			# if the action is computed by s1 and the reward is lower than the average reward
			# s1 gets wrong
			if traj_builders and traj_builders[i]==1:
				self.total_transitions_s1 += 1
				if self.constraints.reward[transition] < -6: #self.getAvgRewardPerTransition():
					self.s1_wrong += 1
					self.s1_wrong_list.append(1)
				else:
					self.s1_wrong_list.append(0)
			else:
				self.total_transitions_s2 += 1
				if self.constraints.reward[transition] < -6: #self.getAvgRewardPerTransition():
					self.s2_wrong += 1
					self.s2_wrong_list.append(1)
				else:
					self.s2_wrong_list.append(0)

			total_reward, total_time = self.getPartialRemainingReward(trajectory, index, time_stat = time_stat)
			# print(state)
			state_s = transition[0]
			action = transition[1]
			state_t = transition[2]

			#compute the coordinates for initial and final state
			state_s_coord = self.grid.world.state_index_to_point(state_s)
			state_t_coord = self.grid.world.state_index_to_point(state_t)

			#update the number of traj for the transition
			temp_ntra_per_transition[state_s_coord][action][state_t_coord] += updateIfZero(temp_ntra_per_transition[state_s_coord][action][state_t_coord])
			if traj_builders: temp_ntra_per_transition_s2[state_s_coord][action][state_t_coord] += updateIfZero(temp_ntra_per_transition_s2[state_s_coord][action][state_t_coord], traj_builders[i])

			#update the number of traj for the (state, action)
			temp_ntra_per_stateAction[state_s_coord][action] += updateIfZero(temp_ntra_per_stateAction[state_s_coord][action])
			if traj_builders: temp_ntra_per_stateAction_s2[state_s_coord][action] += updateIfZero(temp_ntra_per_stateAction_s2[state_s_coord][action], traj_builders[i])

			#update the number of traj for the state
			temp_ntra_per_state[state_s_coord] += updateIfZero(temp_ntra_per_state[state_s_coord])
			if traj_builders: temp_ntra_per_state_s2[state_s_coord] += updateIfZero(temp_ntra_per_state_s2[state_s_coord], traj_builders[i])

			
			#Update the partial length for a given state
			if not self.part_length[state_t_coord]:
				self.part_length[state_t_coord] = []

			#print(f"i: {i}")
			self.part_length[state_t_coord].append(i)

			#Update the partial reward for a given state
			temp_partial_reward += self.constraints.reward[transition]
			self.part_reward[state_t_coord] += temp_partial_reward
			self.part_reward_state_action[state_t_coord][action] += temp_partial_reward

			#create the dictionary for probabilities
			if not self.prob[state_s_coord][action]: 
				self.prob[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			#create the dictionary for probabilities
			if not temp_prob[state_s_coord][action]: 
				temp_prob[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			if total_reward not in temp_prob[state_s_coord][action].keys(): 
				temp_prob[state_s_coord][action][total_reward] = 1

			#create the dictionary for probabilities
			if not self.prob_time[state_s_coord][action]: 
				self.prob_time[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			#create the dictionary for probabilities
			if not temp_prob_time[state_s_coord][action]: 
				temp_prob_time[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			if total_time not in temp_prob_time[state_s_coord][action].keys(): 
				temp_prob_time[state_s_coord][action][total_time] = 1


			if not self.prob_s2[state_s_coord][action]: 
				self.prob_s2[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			#create the dictionary for probabilities
			if not temp_prob_s2[state_s_coord][action]: 
				temp_prob_s2[state_s_coord][action]={}
				#self.prob[state_s_coord][action]['tot_trj'] = 0

			if total_reward not in temp_prob_s2[state_s_coord][action].keys(): 
				temp_prob_s2[state_s_coord][action][total_reward] = 1

			#print(f"i: {i}")
			i += 1
			#self.prob[state_s_coord][action]['tot_trj'] += 1

		#update the number of traj for the terminal state
		temp_ntra_per_state[state_t_coord] += updateIfZero(temp_ntra_per_state[state_t_coord])

		updateDict(temp_prob, self.prob)
		updateDict(temp_prob_s2, self.prob_s2)
		updateDict(temp_prob_time, self.prob_time)

		self.ntra_per_transition  = self.ntra_per_transition  + temp_ntra_per_transition 
		self.ntra_per_stateAction  = self.ntra_per_stateAction  + temp_ntra_per_stateAction 
		self.ntra_per_state  = self.ntra_per_state  + temp_ntra_per_state

		self.ntra_per_transition_s2  = self.ntra_per_transition_s2  + temp_ntra_per_transition_s2
		self.ntra_per_stateAction_s2  = self.ntra_per_stateAction_s2  + temp_ntra_per_stateAction_s2 
		self.ntra_per_state_s2  = self.ntra_per_state_s2  + temp_ntra_per_state_s2

		self.std = np.std(self.ntra_per_transition  / np.sum(self.ntra_per_transition))

	def getRewardS2(self, state_s, verbose=False, maxLike = False):
		"""
		Given:
		states: the actual position in the world

		Compute: 
		reward as the probability that a trajectory from state 'state_s' takes action 'action' 
		confidence as the ratio between the reward and the normalized standard deviation
		"""
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		# total number of traj going through state_s with any action
		total_traj = np.sum(self.ntra_per_stateAction_s2[state_s_coord])
		if total_traj == 0: return -4
		if verbose: print(f"S2 state: {state_s}")
 
		temp_reward_list = []
		temp_prob_list = []
		for action in range(8):
			#compute the list of reward*prob for an action			
			temp_list = self.computeRewardsList(self.ntra_per_stateAction_s2, self.prob_s2, state_s, action)
			
			prob_take_action = self.ntra_per_stateAction_s2[state_s_coord][action] / total_traj
			if verbose: print(f"action:{action} \n temp_list {temp_list} \t prob_take_action {prob_take_action}")
			#if temp_list !=[]: 
				#compute the exp_reward for an action
			temp_reward_list.append(np.sum(temp_list))#*prob_take_action)
			temp_prob_list.append(prob_take_action)
			
		if verbose: print(f"temp_reward_list {temp_reward_list}")
		#exp_reward = computeExpRewardState(self.ntra_per_stateAction_s2, self.ntra_per_state_s2, temp_reward_list, state_s_coord)
		temp_reward_list = np.array(temp_reward_list)
		temp_prob_list = np.array(temp_prob_list)
		if not maxLike: 
			exp_reward = np.sum(temp_reward_list * temp_prob_list)
		else:
			exp_reward = np.argwhere(temp_prob_list == np.amax(temp_prob_list))
			exp_reward = exp_reward.reshape((len(exp_reward),))
			#print(f"exp_reward: {exp_reward}")
			exp_reward = np.random.choice(exp_reward, 1)[0]
			exp_reward = temp_reward_list[exp_reward]

		return exp_reward

	def getReward(self, state_s, action, immediate = False, verbose=False):
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
			return float('-inf'), 1E-03

		#print(f"{self.prob[state_s_coord][action]}")
		if verbose: print(f"IMMEDIATE: {immediate}	S1 state: {state_s} \t action: {action} ")

		'''
		build the distribution of reward * prob where:
			key is the reward and the dictionary value associated to the key is the number of traj
			through state action with that reward
		'''

		temp_ntra_per_stateAction = np.sum(self.ntra_per_stateAction[state_s_coord])
		if immediate:
			temp_list = self.computeRewardsList(self.ntra_per_transition, self.grid.reward, state_s, action, immediate=immediate, verbose=verbose)
		else:
			temp_list = self.computeRewardsList(self.ntra_per_stateAction, self.prob, state_s, action, immediate=immediate, verbose=verbose)
		
		if verbose: print(f"action:{action} \n temp_list {temp_list} \t temp_ntra_per_stateAction {temp_ntra_per_stateAction}")
		exp_reward = np.sum(temp_list)
		#temp_list = temp_list /exp_reward
		#print(f"temp_list_norm: {temp_list/exp_reward}")
		#print(f"state: {state_s} \t action: {action} \t {self.prob[state_s_coord][action]}")
		if verbose: print(f"templist: {temp_list}\n")
		#move_cell_trajectories = self.prob[state_s_coord][action]['tot_trj']
		#print(f"move: {move_cell_trajectories}\t tot_trj {self.prob[state_s_coord][action]['tot_trj']}")
		#trajectories_per_state = self.getNTrajectories(state_s)

		#r = move_cell_trajectories / trajectories_per_state
		#print(f"r_pre: {r}")
		r = self.ntra_per_stateAction[state_s_coord][action] / temp_ntra_per_stateAction
		#print(f"r_post: {r}")
		temp_std = np.std(temp_list)

		if exp_reward == float("-inf"):
			confidence = 1E-03
		else:
			confidence = expit(r/ (temp_std + 1e-10))
		#print(f"confidence: {confidence}")
		#confidence = expit( r / (temp_std + 1e-10))
		#print(f"confidence: {confidence}")
		if verbose: print(f"exp_rew: {exp_reward} \t temp_std: {temp_std} \t r: {r} \t confidence: {confidence}")

		return exp_reward, confidence


	def getExpTime(self, state_s, action, immediate = False, verbose=False):
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
		if not self.prob_time[state_s_coord][action]:
			return float('+inf')

		#print(f"{self.prob[state_s_coord][action]}")
		if verbose: print(f"IMMEDIATE: {immediate}	S1 state: {state_s} \t action: {action} ")

		'''
		build the distribution of reward * prob where:
			key is the reward and the dictionary value associated to the key is the number of traj
			through state action with that reward
		'''

		temp_list = self.computeRewardsList(self.ntra_per_stateAction, self.prob_time, state_s, action, immediate=immediate, verbose=verbose)
		
		if verbose: print(f"action:{action} \n temp_list {temp_list} \t temp_ntra_per_stateAction {temp_ntra_per_stateAction}")
		exp_time = np.sum(temp_list)
		#temp_list = temp_list /exp_reward
		#print(f"temp_list_norm: {temp_list/exp_reward}")
		#print(f"state: {state_s} \t action: {action} \t {self.prob[state_s_coord][action]}")
		if verbose: print(f"templist: {temp_list}\n")
		#move_cell_trajectories = self.prob[state_s_coord][action]['tot_trj']
		#print(f"move: {move_cell_trajectories}\t tot_trj {self.prob[state_s_coord][action]['tot_trj']}")
		#trajectories_per_state = self.getNTrajectories(state_s)

		#r = move_cell_trajectories / trajectories_per_state
		#print(f"r_pre: {r}")
		##r = self.ntra_per_stateAction[state_s_coord][action] / temp_ntra_per_stateAction
		#print(f"r_post: {r}")
		##temp_std = np.std(temp_list)

		if verbose: print(f"exp_time: {exp_time}")

		return exp_time

	def getNTrajectories(self, state_s):
		#return np.sum(self.ntra_per_transition[self.grid.world.state_index_to_point(state_s)])
		return self.ntra_per_state[self.grid.world.state_index_to_point(state_s)]

	def getNTrajectoriesStateAction(self, state_s, action):
		#return np.sum(self.ntra_per_transition[self.grid.world.state_index_to_point(state_s)])
		return self.ntra_per_stateAction[self.grid.world.state_index_to_point(state_s)][action]

	def getAvgPartialReward(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		#print(f"part_reward: {self.part_reward[state_s_coord]} \t tot_traj: {np.sum(self.ntra_per_transition[state_s_coord])}")
		if self.getNTrajectories(state_s) == 0: return 1
		return self.part_reward[state_s_coord] / self.getNTrajectories(state_s)

	def getAvgPartialLength(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		#print(f"part_reward: {self.part_reward[state_s_coord]} \t tot_traj: {np.sum(self.ntra_per_transition[state_s_coord])}")
		if self.part_length[state_s_coord]:
			return float(np.sum(self.part_length[state_s_coord])) / self.getNTrajectories(state_s)
		
		return float('+inf')

	def getAvgRewardPerTransition(self):
		return self.avg_reward / self.total_transitions

	def getM(self):
		if  self.total_transitions_s1 ==0 : return 0
		#return self.s1_wrong / self.total_transitions
		return np.sum(self.s1_wrong_list[-201:-1]) / len(self.s1_wrong_list[-201:-1])

	def getM2(self):
		if  len(self.s2_wrong_list[-201:-1]) ==0 : return 0
		#return self.s1_wrong / self.total_transitions
		return np.sum(self.s2_wrong_list[-201:-1]) / len(self.s2_wrong_list[-201:-1])

	def getAvgPartialRewardStateAction(self, state_s, action):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		return self.part_reward_state_action[state_s_coord][action] / self.getNTrajectoriesStateAction(state_s, action)

	def getMinMaxPartialReward(self, state_s, s2 = False):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		min_rew = float('+inf')
		max_rew = float('-inf')
		temp_prob = self.prob
		if s2: temp_prob = self.prob_s2

		#iterate along the dict where keys are reward values
		for action in temp_prob[state_s_coord]:
			#print(f"action {action}")
			if action:
				min_rew = min(min_rew,min(action.keys()))
				max_rew = max(max_rew,max(action.keys()))

		return min_rew, max_rew

	def getMinMaxPartialLength(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		min_rew = float('+inf')
		max_rew = float('-inf')
		
		#if s2: temp_prob = self.prob_s2
		#print(f"MInMAX for state {state_s} on {self.part_length[state_s_coord]}")
		min_rew = np.min(self.part_length[state_s_coord])
		max_rew = np.max(self.part_length[state_s_coord])

		return min_rew, max_rew

	def getNTrajectoryStateS2(self, state_s):
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		return self.ntra_per_state_s2[state_s_coord]

	def computeRewardsList(self, ntra_per_stateAction, prob, state_s, action, verbose=False, immediate = False):
		temp_list = []
		state_s_coord = self.grid.world.state_index_to_point(state_s)
		if np.sum(ntra_per_stateAction[state_s_coord][action])>0:
			if immediate:
				#compute the total number of trajectories per (state,action)
				total_traj = np.sum(ntra_per_stateAction[state_s_coord][action])
				#print(f"IMMEDIATE \t total_traj: {total_traj}")
				#print(f"state_s {state_s} \t action {action} \t {ntra_per_stateAction[state_s_coord][action]}")
				#print(f"prob {prob[state_s][action]}")
				#for immediate ntra_per_stateAction == ntra_per_transition and prob == grid.reward
				temp_list = ntra_per_stateAction[state_s_coord][action] * prob[state_s][action].reshape((self.grid.world.size,self.grid.world.size))
				temp_list = temp_list / total_traj
				#print(f"temp_list: {temp_list}")
			else:
				total_traj = 0
				#Compute the total number of trajectories going trough state taking action
				for key in prob[state_s_coord][action].keys():
					total_traj += prob[state_s_coord][action][key]

				for key in prob[state_s_coord][action].keys():
					'''
					key represents a reward
					for each key, the value is the number of trajectories with that reward
					'''
					if key is not 'tot_trj':
						temp_reward = key
						temp_prob = prob[state_s_coord][action][key] / total_traj
						temp_list.append(temp_reward * temp_prob)
						if verbose: print(f"temp_reward * temp_prob {temp_reward , temp_prob} \t {prob[state_s_coord][action][key] , total_traj}")
		if verbose: print(f"temp_list {temp_list}\n")
		return temp_list

def computeExpRewardState(ntra_per_stateAction, ntra_per_state, exp_rew_per_action, state_s_coord):
	temp_list = []
	i = 0
	for rew in exp_rew_per_action:
		if rew:
			temp_prob = ntra_per_stateAction[state_s_coord][i] / ntra_per_state[state_s_coord]
			temp_list.append(rew * temp_prob)
		i += 1
	return np.sum(temp_list)

def updateIfZero(number, condition=0):
	if number==0 and condition==0: 
		return 1
	return 0

def updateDict(source, dest):
	#populate le dictionary, each reward is a key and the value is the number of trajectory 
	# passing through state_s, action with that reward
	for x in range(source.shape[0]):
		for y in range(source.shape[1]):	
			for d in range(source.shape[2]):
				if source[x][y][d]:
					for key in source[x][y][d].keys():
						if key not in dest[x,y][d].keys(): 
							dest[x,y][d][key] = source[x][y][d][key]
						else: 
							dest[x,y][d][key] += source[x][y][d][key]