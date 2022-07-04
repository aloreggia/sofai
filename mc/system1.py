from tabnanny import verbose
import numpy as np
from random import random

class System1Solver:
	def __init__(self, myopic=False, random=False, ra=0.0):
		super()
		self.myopic = myopic
		self.random = random
		self.ra = ra

	def policy(self, modelSelf, state):

		#per ogni azione computa la probabilità che appartenga a una traiettoria positiva
		#argmax prob dell'azione 

		random_chance = random()

		if random_chance > self.ra or self.random == True:
			return np.random.choice(range(8)),0

		list_Action = []
		list_Confidence = []

		for a in range(8):
			reward, confidence = modelSelf.getReward(state, a, immediate = self.myopic, verbose=False)
			list_Action.append(reward)
			list_Confidence.append(confidence)

		'''list_Action = [modelSelf.getReward(state, a, immediate = self.myopic)[0]*modelSelf.getReward(state, a, immediate = self.myopic)[1] for a in range(8)]
		#list_Action = [modelSelf.getReward(state, a, immediate = self.myopic)[0] for a in range(8)]
		list_Confidence = [modelSelf.getReward(state, a, immediate = self.myopic)[1] for a in range(8)]'''

		#extract all the action with higher exp * confidence
		#print(f"list_Action: {list_Action}")
		action_max = np.argwhere(list_Action == np.amax(list_Action))
		action_max=action_max.reshape((len(action_max),))
		#exctract a random action among all the ones with higher value
		#print(f"action_max: {action_max}")
		action = np.random.choice(action_max, 1)[0]
		confidence = list_Confidence[action]
		#print(f"action_max: {action} \t confidence {confidence}")

		return action, confidence