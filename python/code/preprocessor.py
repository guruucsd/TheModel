import numpy as np
import scipy.misc
import scipy.ndimage
import os
import PIL
from PIL import Image
import math
import cmath
import matplotlib.pyplot as plt

import importer

class preprocessorClass:
	"""
	preprocesses image

	@author Davis Liang
	@version 1.0
	@date August 6, 2015

	"""

	def __init__(self):
		print('initializing preprocessor')
		
	def filterData(self, image, gaborDictionary):
		"""
		filters data with a set of gabors of various sizes and orientations...

		type image: 50x50 ndarray
		param image: a greyscaled and downsampled image of size 50x50

		type gaborDictionary: 50x50 x numScales x numOrientations list. Each list element is a 50x50 ndarray (single gabor filter)
		param gaborDictionary: a set of all gabor filters (50x50) of various scales and orientations.
		"""

		numScales = gaborDictionary.shape[2]	#extract number of scales from gabor Dictionary
		numOrientations = gaborDictionary.shape[3]	#extract number of orientations

		filteredImage = np.empty([image.shape[0],image.shape[1],numScales, numOrientations], dtype=complex)	#initialize the filtered image ndarray.

		for i in range(numScales):
			for k in range(numOrientations):
				#filter image with gabors of all scales and orientations
				real = scipy.ndimage.convolve(image, gaborDictionary[:,:,i,k].real)	#python allows us to only do this with real numbers?
				imag = scipy.ndimage.convolve(image, gaborDictionary[:,:,i,k].imag)
				filteredImage[:,:,i,k] = real + 1j * image 	#hack-ish way to filter with complex numbers.

		return filteredImage	#return the filtered image

	def createConvFilterBank(self, numScales, numOrientations, gaborSize=25, frequency=1, std_dev=math.pi):
		"""
		creates a gabor filter dictionary

		type numScales: int
		param numScales: number of scales of gabors. I use 5.

		type numOrientations: int
		param numOrientations: number of orientations of gabors. I use 8.

		"""
		print('    building gabors...')
		scale = np.zeros([1,numScales])	#initialize a vector of scales
		orientation = np.zeros([1,numOrientations])	#initialize a vector of orientations

		for i in range(numScales):
			scale[0,i] = (2*math.pi/gaborSize)*(2**(i+1))	#fill the scale vector

		for i in range(numOrientations):
			orientation[0,i] = (math.pi/numOrientations)*(i)	#fill the orientation vector

		#construct your gabors...
		carrier = np.zeros([2*gaborSize, 2*gaborSize], dtype=complex)	
		envelop = np.zeros([2*gaborSize, 2*gaborSize], dtype=complex)
		gabor = np.zeros([2*gaborSize, 2*gaborSize, numOrientations, numScales],dtype=complex)


		for i in range(numScales):
			for k in range(numOrientations):
				for y in range(-gaborSize+1, gaborSize):
					for x in range(-gaborSize+1, gaborSize):
						#comparable to Panqu's version of the Model
						carrier[y+gaborSize-1, x+gaborSize-1] = np.exp(1j*(scale[0,i] * math.cos(orientation[0,k])*y + scale[0,i]*math.sin(orientation[0,k])*x))
						envelop[y+gaborSize-1, x+gaborSize-1] = np.exp(-(scale[0,i]**2*(y**2+x**2))/(2*std_dev*std_dev*frequency))
						gabor[y+gaborSize-1, x+gaborSize-1, i, k] = carrier[y+gaborSize-1, x+gaborSize-1] * envelop[y+gaborSize-1, x+gaborSize-1]

		# your gabors are now built...
		print('    gabors have been successfully built')
		return gabor #return the gabor dictionary.


	#def PCA(self, trainSet):

		

	
if __name__ == '__main__':
	pp = preprocessorClass()
	gaborDict = pp.createConvFilterBank(5,8)
	#import an image
	#then, filter it and see what the dimensionality is, if it makes sense, and how we would split that up
	
	dataPath = "/Users/Davis/Desktop/theModel/data/dummy"
	imp1 = importer.importerClass()
	x = imp1.load(dataPath)

	imData = x[0]
	imName = x[1]
	imLabel = x[2]

	filtIm = pp.filterData(imData[0], gaborDict)

	#imgplot = plt.imshow(imData[0])
	#plt.show()

	#imgplot = plt.imshow((gaborDict[:,:,0,3]).real)
	#plt.show()

	#imgplot = plt.imshow(filtIm[:,:,3,0].real)
	#plt.show()
	#gaborDict[:,:,orientation,size]
	#filtIm[:,:,size, orientation]




