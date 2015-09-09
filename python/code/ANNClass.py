import numpy as np

class ANNClass:
	""" 
	a generalized sigmoid neural network with 3 layers 

	@author Davis Liang
	@version 1.0
	@date August 6, 2015

	"""


	def __init__(self, inpt, labels, numhid):
		"""
		initializes the network with data from the preprocessed datassets and the number of hidden units.

		type inpt: dxi ndarray
		param inpt: each column of inpt corresponds to a different image. The rows just hold data for each image.

		type labels: dxi list
		param labels: you can access the label for image i at label[i]

		type numhid: int
		param numhid: number of hidden units in the network.
		"""
		print('initializing network class... ')
		self.dataset = inpt	#dataset includes the inputs organized by columns. inpt[:,i] corresponds to data point i.
		self.datatarg = labels	#the label for each input
		self.numhid = numhid	#number of hidden units in this network/

		#this code is for the edge case as ndarray vectors are shaped (n,) not (n,1)
		if len(labels.shape) != 1:
			targlen = labels.shape[0]
		else:
			targlen = 1

		#initialize your weight arrays for input to hidden as well as hidden to output.
		self.whi = 2*np.random.rand(self.numhid, inpt.shape[0]+1) - 1 	#input to hidden weights
		self.woh = 2*np.random.rand(targlen, numhid+1) - 1 				#hidden to output weights

		#this will hold the entire error vector in case you want to plot it later in a line graph
		self.Error = []


	def train(self, numIter, learnih, learnho):
		"""
		trains a 3 layer network with sigmoid activations

		type numIter: int
		param numIter: number of epochs to run before stopping

		type learnih: float
		param learnih: input to hidden unit learning rate

		type learnho: float
		param learnho: hidden to output learning rate
		"""
		run = True	#network runs until this is false
		epoch = 0 	#current epoch
		inlen = self.dataset.shape[0] 	#length of input to construct weight matrix
		#self.numhid 

		#for edge case because ndarray column vectors have shape [n,] rather than [n,1]
		if len(labels.shape) != 1: 
			targlen = labels.shape[0]
		else:
			targlen = 1

		#size of dataset
		setSize = self.dataset.shape[1]

		#alpha for momentum
		a = 0.9

		dwoldh = np.zeros([self.numhid, inlen+1])	#previous input to hidden weight change (for momentum)
		dwoldo = np.zeros([targlen, self.numhid+1])	#previous hidden to output weight change
		dwh = np.zeros([self.numhid, inlen+1])	#input to hidden weights, including bias
		dwo = np.zeros([targlen,numhid+1])		#hidden to output weights, including bias

		self.Error = np.empty([1, numIter]) 	#error vector to plot later (if you want)
		print('training... ')

		while (run):
			epoch += 1
			trainError = 0 	#training error for this particular epoch begins at zero.
			if(epoch%10 == 0):
				print('epoch {}... ').format(epoch)

			for i in np.random.permutation(setSize-1): #SGD
				inp = np.append([1], self.dataset[:,i])
				inp = inp.reshape(3,1)

				targ = self.datatarg[i]
				neti = self.whi.dot(inp)

				hout = 1/(1+np.exp(-neti))
				h_layer = np.append(1, hout).reshape(self.numhid+1, 1)
				neto = self.woh.dot(h_layer)
				out = 1/(1+np.exp(-neto))

				oprime = np.multiply(out, 1-out)	#f'(net)
				hprime = np.multiply(hout, 1-hout)

				deltao = np.multiply((targ-out), oprime)	#backpropagated errors.
				deltah = np.multiply(hprime, self.woh[:,1:numhid+1].T*deltao)

				dwo = learnho*(deltao.dot(h_layer.T)) + dwoldo*a 	#weight change
				dwh = learnih*(deltah.dot(inp.T)) + dwoldh*a

				dwoldh = dwh
				self.whi = self.whi + dwh

				dwoldo = dwo
				self.woh = self.woh + dwo

				trainError = trainError + 0.5*sum((targ-out)**2)/(setSize*targlen) 	#sum squared error

			self.Error[0,epoch-1] = trainError
			if(epoch%10 == 0): 
				print('    train error: {}').format(trainError)
			if(epoch == numIter):
				run = False
		print('training ends.')


if __name__ == '__main__':
	print('running XOR experiment... ')
	inpt = np.array([[0,0,1,1],[0,1,0,1]])
	labels = np.array([0,1,1,0])
	numhid = 10
	demoNet = ANNClass(inpt, labels, numhid)
	demoNet.train(100, 0.05, 0.001)

	print('whi: {}').format(demoNet.whi)
	print('woh: {}').format(demoNet.woh)


		




