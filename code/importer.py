import numpy as np
import scipy.misc
import os
import PIL
from PIL import Image



class importerClass:
	""" 
	imports images (and then greyscales) as regular matrices 

	@author Davis Liang
	@version 1.0
	@date August 6, 2015

	"""

	def __init__(self):
		print('setting up importer for the first time')
		
		
	def load(self, path):
		"""
		loads all the images from a directory in path

		type path: string
		param path: the directory of the image dataset. E.G. "/Users/Davis/Desktop/theModel/data/dummy"
		"""
		print('importing images')
		listing = os.listdir(path)		#cd to that directory

		imData = [None]*len(listing)	#initialize empty list for data
		imName = [None]*len(listing)	#names of images
		imLabel = [None]*len(listing)	#image label

		counter = 0	#for terminal output

		for filename in listing:	
			print('    processing image {}...').format(counter+1)	

			impath = os.path.join(path, filename)
			imraw = scipy.misc.imread(impath)	#extracts ndarray 

			#if RGB, greyscale it.
			if(len(imraw.shape) == 3):
		 		imgray = rgb2gray(imraw)
		 	else:
		 		imgray=imraw
		 	
		 	im = scipy.misc.imresize(imgray, (50,50)) #resize data for faster filtering.
		 	imData[counter] = im
		 	imName[counter] = filename
		 	imLabel[counter] = filename[0]

		 	counter += 1

		return [imData, imName, imLabel]

def rgb2gray(rgb):
	"""
	Uses a similar algorithm to rgb2gray in matlab.
	
	type rgb: NxNx3 ndarray
	param rgb: an rgb image with type ndarray 
	"""
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	return gray


if __name__ == '__main__':
	print('loading dummy set... ')
	dataPath = "/Users/Davis/Desktop/theModel/data/dummy"

	imp1 = importerClass()
	x = imp1.load(dataPath)
	imData = x[0]
	imName = x[1]
	imLabel = x[2]

	print(len(imData))
	print(len(imName))
	print(imLabel)
	#listing = os.listdir(path)
