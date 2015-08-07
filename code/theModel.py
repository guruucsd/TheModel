
import importer
import preprocessor
import ANNClass
class theModel:
	""" 
	the main class for running Gary Cottrell's The Model

	@author Davis Liang
	@version 1.0
	@date August 6, 2015

	"""


	def __init__(self):
		print('Initializing The Model')


if __name__ == '__main__':

	dataPath = "/Users/Davis/Desktop/theModel/data/dummy"	#setting up dataPath to dataset directory

	imp1 = importer.importerClass()	#initializing a new importer class to import images
	x = imp1.load(dataPath) #import all images from the directory dataPath
	imData = x[0]	#holds all the images in the dataset. imData[i] corresponds to image i.
	imName = x[1]	#holds the name of the image. imName[i] corresponds the name of image i. See how the names are formatted to extract label data.
	imLabel = x[2]	#holds all the labeling data to be used in the network. imLabel[i] corresponds to labeling for image i.

	pp = preprocessor.preprocessorClass() #initialize a preprocessor class
	gaborDict = pp.createConvFilterBank(5,8)	#build your gabor dictionaries

	filtData = [None]*len(imData) #initialize an array to hold the filtered Data

	#filter your entire dataset
	for i in range(len(imData)):
		print('    filtering image {}').format(i+1)
		filtData[i] = pp.filterData(imData[i], gaborDict)
		
	#filtData[i] corresponds to all the filtered data for image i
	#filtData[i] is organized with shape [50,50,size, orientation] 
	#where each feature map is 50x50 and size and orientation correspond to the gabor that filtered the image.


