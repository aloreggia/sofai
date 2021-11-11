import numpy as np

class System1Solver:
	def __init__(self):
		super()

	def policy(self, modelSelf, state):

		#per ogni azione computa la probabilità che appartenga a una traiettoria positiva
		#argmax prob dell'azione 

		list_Action = [modelSelf.getReward(state, a)[0]*modelSelf.getReward(state, a)[1] for a in range(8)]
		list_Confidence = [modelSelf.getReward(state, a)[1] for a in range(8)]

		#extract all the action with higher exp * confidence
		#print(f"list_Action: {list_Action}")
		action_max = np.argwhere(list_Action == np.amax(list_Action))
		action_max=action_max.reshape((len(action_max),))
		#exctract a random action among all the ones with higher value
		#print(f"action_max: {action_max}")
		action = np.random.choice(action_max, 1)[0]
		confidence = list_Confidence[action]

		return action, confidence