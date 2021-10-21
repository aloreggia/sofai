import numpy as np
from max_ent.gridworld.trajectory import Trajectory
from max_ent.algorithms.gridworld_icrl import Demonstration
import time
from mc.self import *

class MCA:

	def __init__(self, s1, s2, modelSelf, threshold1=0, threshold2 = 0.5, \
				threshold3 = 0.5, initial_time=100, threshold4=0.5, \
				threshold5 = 0, only_s1 = False, only_s2 = False):
		assert(only_s1 != only_s2 or (not only_s1 and not only_s2 ))
		self.s1 = s1
		self.s2 = s2
		self.modelSelf = modelSelf
		self.threshold1 = threshold1
		self.threshold2 = threshold2
		self.threshold3 = threshold3
		self.threshold4 = threshold4
		self.threshold5 = threshold5
		self.usage_s1 = 0
		self.usage_s2 = 0
		self.time_usage_s2 = 0
		self.w=[1.0, 0.0]
		self.trajectory_stat = []
		self.time_left = initial_time * 1000 #change sec in millisec
		self.only_s1 = only_s1
		self.only_s2 = only_s2

	def generate_trajectory(self, max_len=200):
		"""
		Generate a single trajectory.
		Args:
			world: The world for which the trajectory should be generated.
			policy: A function (state: Integer) -> (action: Integer) mapping a
				state to an action, specifying which action to take in which
				state. This function may return different actions for multiple
				invokations with the same state, i.e. it may make a
				probabilistic decision and will be invoked anew every time a
				(new or old) state is visited (again).
			start: The starting state (as Integer index).
			final: A collection of terminal states. If a trajectory reaches a
				terminal state, generation is complete and the trajectory is
				returned.
		Returns:
			A generated Trajectory instance adhering to the given arguments.
		"""

		state = self.modelSelf.getStart()
		final = self.modelSelf.getGoal()

		trajectory = []
		current_reward = 0 
		trial = 0
		#each component corresponds to a transition in the trajectory
		# 0 means action computed with s2
		# 1 means action computed with s1
		trajectory_builder = []
		while state != final:
			s1_use = 1
			engageS2 = False
			if len(trajectory) > max_len:  # Reset and create a new trajectory
				if trial >= 5:
					print('Warning: terminated trajectory generation due to unreachable final state.')
					return Trajectory(trajectory), False, trajectory_builder    #break
				trajectory = []
				state = self.modelSelf.getStart()
				trial += 1

			#S1 computes an action based on previous experience
			if not self.only_s2:
				time_exp = int(round(time.time() * 1000))
				action, confidence = self.s1.policy(self.modelSelf, state)
				time_exp = int(round(time.time() * 1000)) - time_exp
			else:
				confidence = 1
				action = 0

			expected_avg_reward = self.modelSelf.getAvgPartialReward(state)

			#check whether the system has enough time to run system 2
			if not self.only_s1:
				if self.only_s2 or self.time_left >= self.threshold4:
					# Check second condition: if we should consider action then 
					# expected_reward = current_reward + exp_reward(action) but then what is the comparison with?
					# else if action is not involved then this is not a condition about s1 or s2 based on action	
					if self.only_s2 or (self.modelSelf.getNTrajectories(state) <= self.threshold1) or (expected_avg_reward - current_reward > self.threshold2) or (confidence <= self.threshold3):
						engageS2 = False

						min_rew, max_rew = self.modelSelf.getMinMaxPartialReward(state)
						min_rew = np.abs(current_reward - min_rew)
						max_rew = np.abs(current_reward - max_rew)
						max_diff_rew = max(min_rew, max_rew)
						delta_reward = np.abs(current_reward - expected_avg_reward)/max_diff_rew

						expected_rew_move_s1 = current_reward + self.modelSelf.getAvgPartialRewardStateAction(state,action)

						expected_cost_s2 = 1 #should de derived from past experience
						if self.usage_s2 != 0:
							expected_cost_s2 = (self.time_usage_s2 / self.usage_s2) #compute avg S2 time
							expected_cost_s2 /= self.time_left 

						if self.only_s2 or (delta_reward / expected_cost_s2) > self.threshold4:
							engageS2 = True
							#adjust w based on how the agent behaves so far
							x = 0.0
							if expected_avg_reward > current_reward:
								x = np.abs(current_reward - expected_avg_reward)/max_diff_rew

							x = min(1.0,x)
							w = [1-x, x]
							if self.threshold5 == 1: 
								w = [x, 1-x]

							time_exp = int(round(time.time() * 1000))
							#print(f"{w}")
							action = self.s2.policy(self.modelSelf, state, w)
							time_exp = int(round(time.time() * 1000)) - time_exp

			next_s = range(self.modelSelf.getNStates())
			next_p = self.modelSelf.getWorld().p_transition[state, :, action]

			next_state = np.random.choice(next_s, p=next_p)

			transition = (state, action, next_state)
			trajectory.append(transition)
			current_reward += self.modelSelf.constraints.reward[transition]

			state = next_state

			if not engageS2:
				self.usage_s1 += 1
				s1_use = 1
			else:
				self.usage_s2 += 1
				self.time_usage_s2 += time_exp
				s1_use = 0
				self.time_left -= time_exp #reduce the remaining time
			
			trajectory_builder.append(s1_use)

		
		return Trajectory(trajectory), True, trajectory_builder


	def generate_trajectories(self, n, discard_not_feasable=False):
		"""
		Generate multiple trajectories.
		Args:
			n: The number of trajectories to generate.
			world: The world for which the trajectories should be generated.
			policy: A function `(state: Integer) -> action: Integer` mapping a
				state to an action, specifying which action to take in which
				state. This function may return different actions for multiple
				invokations with the same state, i.e. it may make a
				probabilistic decision and will be invoked anew every time a
				(new or old) state is visited (again).
			start: The starting state (as Integer index), a list of starting
				states (with uniform probability), or a list of starting state
				probabilities, mapping each state to a probability. Iff the
				length of the provided list is equal to the number of states, it
				is assumed to be a probability distribution over all states.
				Otherwise it is assumed to be a list containing all starting
				state indices, an individual state is then chosen uniformly.
			final: A collection of terminal states. If a trajectory reaches a
				terminal state, generation is complete and the trajectory is
				complete.
			discard_not_feasable: Discard trajectories that not reaching the 
				final state(s)
		Returns:
			A generator expression generating `n` `Trajectory` instances
			adhering to the given arguments.
		"""
		world = self.modelSelf.getWorld()
		start = self.modelSelf.getStart() 
		final = self.modelSelf.getGoal()

		start_states = np.atleast_1d(start)

		def _generate_one():
			if len(start_states) == world.n_states:
				s = np.random.choice(range(world.n_states), p=start_states)
			else:
				s = np.random.choice(start_states)

			return self.generate_trajectory()

		list_tr = []
		for _ in range(n):
			tr, reachable, temp_stats = _generate_one()
			if reachable or not discard_not_feasable:
				self.modelSelf.updateModel(tr)
				list_tr.append(tr)
				self.trajectory_stat.append(temp_stats)
		
		return Demonstration(list_tr, self.s1.policy)