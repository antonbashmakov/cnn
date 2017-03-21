import gzip
import pickle

import numpy as np


def loadData() : 

	f = gzip.open('data/mnist.pkl.gz', 'rb')

	trainingData, validationData, testData = pickle.load(f, encoding = 'latin1')

	f.close()

	return (trainingData, validationData, testData) 

def loadDataWrapper() : 

	''' 
		Convert loadData() result to more appropriate format
	'''

	trD, vaD, teD = loadData()

	trainingInputs = [np.reshape(x, (784, 1)) for x in trD[0]]
	trainingResults = [vectorizedResult(y) for y in trD[1]]

	trainingData = zip(trainingInputs, trainingResults)

	validationInputs = [np.reshape(x, (784, 1)) for x in vaD[0]]
	
	validationData = zip(validationInputs, vaD[1])

	testInputs = [np.reshape(x, (784, 1)) for x in teD[0]]

	testData = zip(testInputs, teD[1])

	return (trainingData, validationData, testData)


def vectorizedResult(j) :

	e = np.zeros((10, 1))

	e[j] = 1.0

	return e