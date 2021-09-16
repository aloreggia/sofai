import numpy as np
from max_ent.gridworld.trajectory import Trajectory
from max_ent.algorithms.gridworld_icrl import Demonstration

import sys, os
sys.path.append(os.path.abspath(os.path.join('../')))

class MCA:

	def __init__(self, s1, modelSelf, n_tra_threshold=0):
		self.s1 = s1
		self.modelSelf = modelSelf
		self.n_tra_threshold = n_tra_threshold
		self.usage_s1 = 0
		self.usage_s2 = 0

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
		engageS2 = False

		trajectory = []
		trial = 0
		while state != final:
			if len(trajectory) > max_len:  # Reset and create a new trajectory
				if trial >= 5:
					print('Warning: terminated trajectory generation due to unreachable final state.')
					return Trajectory(trajectory), False    #break
				trajectory = []
				state = self.modelSelf.getStart()
				trial += 1

			action = self.s1.policy(self.modelSelf, state)

			if self.modelSelf.getNTrajectories(state) <= self.n_tra_threshold:
				engageS2 = True
				self.usage_s2 += 1


			next_s = range(self.modelSelf.getNStates())
			next_p = self.modelSelf.getWorld().p_transition[state, :, action]

			next_state = np.random.choice(next_s, p=next_p)

			trajectory.append((state, action, next_state))
			state = next_state

			if not engageS2:
				self.usage_s1 += 1

		return Trajectory(trajectory), True


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
			tr, reachable = _generate_one()
			if reachable or not discard_not_feasable:
				self.modelSelf.updateModel(tr)
				list_tr.append(tr)
		
		return Demonstration(list_tr, self.s1.policy)