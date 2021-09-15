import numpy as np

class System1Solver:
	def __init__(self):
		super()

	def policy(self, modelSelf, state):

		#per ogni azione computa la probabilità che appartenga a una traiettoria positiva
		#argmax prob dell'azione 

		action = np.argmax([modelSelf.getReward(state, a)[0]*modelSelf.getReward(state, a)[1] for a in range(8)])

		return action