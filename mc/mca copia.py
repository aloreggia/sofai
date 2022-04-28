import numpy as np
from max_ent.gridworld.trajectory import Trajectory
from max_ent.algorithms.gridworld_icrl import Demonstration
import time
from mc.self import *
from mc.system1 import *
from mc.system2 import *

from random import random

class MCA:

	def __init__(self, n=None, c=None, demo=None, s1=None, s2=None, modelSelf=None, threshold1=200, threshold2 = 0.8, \
				threshold3 = 0.5, initial_time=100, threshold4 = 0, \
				threshold5 = 0, threshold6 = 1, threshold7 = 0.5, only_s1 = False, only_s2 = False, mixed= False):
		assert(only_s1 != only_s2 or (not only_s1 and not only_s2 ))

		if s1 != None: self.s1 = s1
		else: self.s1 = System1Solver()

		if s2 != None: self.s2 = s2
		else: self.s2 = System2Solver()

		if modelSelf!= None: self.modelSelf = modelSelf
		else: self.modelSelf = ModelSelf(n, c, demo)

		print(f"threshold1: {threshold1}")
		self.threshold1 = threshold1
		self.threshold2 = threshold2
		self.threshold3 = threshold3
		self.threshold4 = threshold4
		self.threshold5 = threshold5
		self.threshold6 = threshold6
		self.threshold7 = threshold7
		self.usage_s1 = 0
		self.usage_s2 = 0
		self.time_usage_s2 = 0
		#set w based on the type
		if self.threshold5 == 0: 
			self.w = [1, 0]
		elif self.threshold5 == 1: 
			self.w = [0, 1]
		elif self.threshold5 == 2:
			self.w = [0.5, 0.5]

		self.trajectory_stat = []
		self.thresholds_stat = []
		self.thresholds_mask = []

		self.time_stat = []
		self.violations = []
		self.action_reward = []
		self.fixed_time_left = initial_time * 1000 #change sec in millisec
		self.time_left = initial_time * 1000 #change sec in millisec
		self.only_s1 = only_s1
		self.only_s2 = only_s2

		self.mixed = mixed
		if self.mixed: 
			self.only_s1 = False
			self.only_s2 = True

	def getStatistics(self, verbose=False):
		'''
		Return the average amount of times that s1 and s2 are used per-trajectory
		'''
		percentage_s1 = 0.0
		percentage_s2 = 0.0

		for trajectory, time_spent in zip(self.trajectory_stat, self.time_stat):
			percentage_s1 += np.sum(trajectory)/len(trajectory)
			percentage_s2 += (len(trajectory) - np.sum(trajectory))/len(trajectory)

		if verbose: print(f"{percentage_s1/len(self.trajectory_stat):.4f}, {percentage_s2/len(self.trajectory_stat):.4f}")
		return percentage_s1/len(self.trajectory_stat), percentage_s2/len(self.trajectory_stat)

	def generate_trajectory(self, max_length = 200):
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
		trajectory_builder = []
		time_builder = []
		viol = []
		act_reward =[]
		temp_thresholds_stat = []
		temp_thresholds_mask = []
		self.time_left = self.fixed_time_left
		
		current_reward = 0 
		trial = 0
		#each component corresponds to a transition in the trajectory
		# 0 means action computed with s2
		# 1 means action computed with s1
		
		while state != final:
			s1_use = 1
			engageS2 = False
			action_thresholds = np.zeros(7)
			action_thresholds_mask = np.zeros(7)
			if len(trajectory) > max_length:  # Reset and create a new trajectory
				if trial >= 5:
					print('Warning: terminated trajectory generation due to unreachable final state.')
					return Trajectory(trajectory), False, trajectory_builder, time_builder, act_reward, temp_thresholds_stat, temp_thresholds_mask
				trajectory = []
				trajectory_builder = []
				time_builder = []
				viol = []
				act_reward =[]
				current_reward = 0
				state = self.modelSelf.getStart()
				trial += 1
				temp_thresholds_stat = []
				self.time_left = self.fixed_time_left

			#S1 computes an action based on previous experience
			if not self.only_s2:
				time_exp = int(round(time.time() * 1000))
				action, confidence = self.s1.policy(self.modelSelf, state)
				time_exp = int(round(time.time() * 1000)) - time_exp
			else:
				engageS2 = True
				confidence = 1
				action = 0

			expected_avg_reward = self.modelSelf.getAvgPartialReward(state)

			#check whether the system has enough time to run system 2
			if not self.only_s1:

				#set w based on the type
				w = self.w

				action_thresholds[0] = self.modelSelf.getNTrajectories(state)
				action_thresholds[1] = current_reward / expected_avg_reward
				if self.modelSelf.total_transitions > 200:
					action_thresholds[2] = (1 - self.modelSelf.getM()) * confidence
				else:
					action_thresholds[2] = confidence

				if not self.only_s2 and ((action_thresholds[0] <= self.threshold1) or (action_thresholds[1] < self.threshold2) or (action_thresholds[2] <= self.threshold3)):
					engageS2 = False
					#print(f"self.modelSelf.getM(): {self.modelSelf.getM()}")

					if (action_thresholds[0] <= self.threshold1):
						action_thresholds_mask[0] = 1 
					if (action_thresholds[1] < self.threshold2):
						action_thresholds_mask[1] = 1  
					if (action_thresholds[2] <= self.threshold3):
						action_thresholds_mask[2] = 1  					

					action_thresholds[5] = self.modelSelf.getNTrajectoryStateS2(state)
					
					if self.modelSelf.getNTrajectoryStateS2(state)< self.threshold6:
						action_thresholds_mask[5] = 1  					
						#engage S2 at random
						random_chance = random()
						action_thresholds[6] = random_chance

						if  random_chance < self.threshold7: 
							engageS2 = True
							action_thresholds_mask[6] = 1  					
					else:
						min_rew_s2, max_rew_s2 = self.modelSelf.getMinMaxPartialReward(state, s2 = True)
						#print(f"min_rew_s2, max_rew_s2 {min_rew_s2, max_rew_s2}")
						min_rew, max_rew = self.modelSelf.getMinMaxPartialReward(state)
						min_rew = np.abs(min_rew_s2 - max_rew)
						max_rew = np.abs(max_rew_s2 - min_rew)
						max_diff_rew = max(min_rew, max_rew)

						expected_rew_move_s1 = self.modelSelf.getReward(state, action)[0]
						expected_rew_move_s2 = self.modelSelf.getRewardS2(state)
						delta_reward = (expected_rew_move_s2 - expected_rew_move_s1) / max_diff_rew

						#compute the average time taken by S2 to compute a move
						expected_time_s2 = 1 #should de derived from past experience
						if self.usage_s2 != 0:
							expected_time_s2 = (self.time_usage_s2 / self.usage_s2) #compute avg S2 time

						expected_cost_s2 = expected_time_s2 / self.time_left
						#print(f"{expected_cost_s2} and {expected_rew_move_s2} - {expected_rew_move_s1} / {max_diff_rew} / expected_cost_s2 {expected_cost_s2}")
						action_thresholds[3] = (delta_reward / expected_cost_s2)
						action_thresholds[4] = expected_cost_s2
						if (expected_cost_s2 <= 1 and (delta_reward / expected_cost_s2) >= self.threshold4):
							action_thresholds_mask[3] = 1  					
							engageS2 = True
							#print("Engage S2")
						#else:
							#self.modelSelf.getRewardS2(state, verbose=True)
							#self.modelSelf.getReward(state, action, verbose=True)
							#print(f"{expected_rew_move_s2} - {expected_rew_move_s1} / {max_diff_rew} / expected_cost_s2 {expected_cost_s2}")

				#if self.only_s2: print(f"Engage S2: {engageS2}")
				if engageS2:
					action_thresholds_mask[4] = 1  					
					if self.threshold5 == 0: 
						x = 0.0
						if expected_avg_reward >= current_reward:
							min_rew, max_rew = self.modelSelf.getMinMaxPartialReward(state)
							if min_rew != None and max_rew != None:
								min_rew = np.abs(current_reward - min_rew)
								max_rew = np.abs(current_reward - max_rew)
								max_diff_rew = max(min_rew, max_rew)
								x = np.abs(current_reward - expected_avg_reward) / max_diff_rew
						x = min(1.0,x)
						w = [1-x, x]
					elif self.threshold5 == 1: 
						x = 0.0
						expected_avg_length = self.modelSelf.getAvgPartialLength(state) 
						#print(f"{expected_avg_length}")
						current_length = len(trajectory_builder)
						if expected_avg_length >= current_length:
							min_len, max_len = self.modelSelf.getMinMaxPartialLength(state)
							if min_len != None and max_len != None:
								min_len = np.abs(current_length - min_len)
								max_len = np.abs(current_length - max_len)
								max_diff_len = max(min_len, max_len)
								x = np.abs(current_length - expected_avg_length) / max_diff_len
						x = min(1.0,x)
						w = [x, 1-x]
					elif self.threshold5 == 2:
						x = 0.0
						min_rew, max_rew = self.modelSelf.getMinMaxPartialReward(state)
						if min_rew != None and max_rew != None:
							min_rew = np.abs(current_reward - min_rew)
							max_rew = np.abs(current_reward - max_rew)
							max_diff_rew = max(min_rew, max_rew)
							x = (current_reward - expected_avg_reward) / max_diff_rew
							x = min(1.0,x)
							x = max(-1.0, x)
						if x>=0: w = np.array([1.0 +x, 1.0 -x])/2
						else: w = np.array([1.0 +x, 1.0 -x])/2
						#print(f"{w} {x}")

					time_exp = int(round(time.time() * 1000))
					#print(f"{w} {x}")
					action = self.s2.policy(self.modelSelf, state, w)
					time_exp = int(round(time.time() * 1000)) - time_exp


			next_s = range(self.modelSelf.getNStates())
			next_p = self.modelSelf.getWorld().p_transition[state, :, action]

			next_state = np.random.choice(next_s, p=next_p)

			transition = (state, action, next_state)
			trajectory.append(transition)
			current_reward += self.modelSelf.constraints.reward[transition]
			act_reward.append(self.modelSelf.constraints.reward[transition])
			temp_thresholds_stat.append(action_thresholds)
			temp_thresholds_mask.append(action_thresholds_mask)

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
			time_builder.append(time_exp)


		
		return Trajectory(trajectory), True, trajectory_builder, time_builder, act_reward, temp_thresholds_stat, temp_thresholds_mask


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
			if self.mixed and len(self.trajectory_stat)>=200:
				self.only_s1 = True
				self.only_s2 = False	

			#tr, reachable, temp_stats, temp_time_stat, temp_act_reward, temp_thresholds_stat = _generate_one()
			result_traj = _generate_one()
			'''if result_traj[1] or not discard_not_feasable:
				self.modelSelf.updateModel(result_traj[0], result_traj[2])
				list_tr.append(tr)
				self.trajectory_stat.append(temp_stats)
				self.time_stat.append(temp_time_stat)
				self.action_reward.append(temp_act_reward)
				self.thresholds_stat.append(temp_thresholds_stat)'''
			if result_traj[1] or not discard_not_feasable:
				self.modelSelf.updateModel(result_traj[0], result_traj[2])
				list_tr.append(result_traj[0])
				self.trajectory_stat.append(result_traj[2])
				self.time_stat.append(result_traj[3])
				self.action_reward.append(result_traj[4])
				self.thresholds_stat.append(result_traj[5])
				self.thresholds_mask.append(result_traj[6])
		
		return Demonstration(list_tr, self.s1.policy)