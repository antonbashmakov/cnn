
import numpy as np
import random

class Network() : 

	'''
		Cost function constants
	'''
	CROSS_ENTROPY = 'CE'

	'''
    Base network class.
	'''

	def __init__ (self, topology, costFunctionType = None) :



		self.numOfLayers = len(topology)
		self.topology = topology
		self.biases = [np.random.randn(y, 1) for y in topology[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])]

	def sigmoid (self, z) : 
		'''
			Sigmoid function gor our neuron
		'''
		return 1.0 / (1 + np.exp(-z)) 

	def sigmoidPrime (self, z) : 
		return self.sigmoid(z) * ( 1 - self.sigmoid(z))


	def feedForward (self, a) :
		for b, w in zip(self.biases, self,weights) : 
			a = self.sigmoid(np.dot(w, a) + b)

	def updateMiniBatch (self, miniBatch, eta) :
		'''
			Tune the weights for minibatch using gradient descent
		'''
		nablaB = [np.zeros(b.shape) for b in self.biases] 
		nablaW = [np.zeros(w.shape) for w in self.weights]

		for x,y in miniBatch: 
			deltaNablaB, deltaNablaW = self.backprop(x, y)

			nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB) ]
			nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW) ]

		self.weights = [w - (eta / len(miniBatch)) * nw for w, nw in zip(self.weights, nablaW)]
		self.biases = [b - (eta / len(miniBatch)) * nb for b, nb in zip(self.biases, nablaB)]



	def backprop(self, x, y) :
		nablaB = [np.zeros(b.shape) for b in self.biases]
		nablaW = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []

		for b, w in zip(self.biases, self.weights) : 

			z = np.dot(w, activation) + b

			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)

		#backwarding, we are going from the last layer to the first
		delta = self.costDerivative(activations[-1], y) * self.sigmoidPrime(zs[-1]) #calculate direvative for the last layer
		
		nablaB[-1] = delta 
		nablaW[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, self.numOfLayers) : 
			z = zs[-l]
			sp = self.sigmoidPrime(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			nablaB[-l] = delta
			nablaW[-l] = np.dot(delta, activations[-l -1].transpose())

		return (nablaB, nablaW)



	def costDerivative(self, outputActivations, y) :
		return (outputActivations - y)

	def SGD (self, trainingData, epochs, miniBatchSize, eta, testData = None) : 
		'''
			Gradient descent for the network.
			eta - learning rate
		'''

		n = len(testData)

		for j in range(epochs) : 
			random.shuffle(trainingData)

			miniBatches = [trainingData[k : k + miniBatchSize] for k in range(0, n, miniBatchSize)] #form mini batches from the training data

			for miniBatche in miniBatches : 
				self.updateMiniBatch(miniBatche, eta)

			print("Epoch %d complete" % (j))




    