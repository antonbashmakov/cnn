import network
import load_data


if __name__ == "__main__":
	trainingData, validationData, testData = load_data.loadDataWrapper()	


	net = network.Network([784, 30, 10])
	net.SGD(list(trainingData) , 30, 10, 3.0, testData = list(testData))
