 #!/usr/local/bin/python

#########################################################################
# Bag of code generation for each pair of files                         #
# Author: Ashish Arya                                                   #
# Date Created: 14/10/2017                                              #
# Purpose: 1) To generate bag of code from computed clusters            #
#                                                                       #
#                                                                       #
#########################################################################

import numpy as np
import pickle
import os
import sys

################    All Constants and paths used    #####################
path = "./data/"
n_classes = 26 # Number of classes in bird data
relativePathForTrain = "./data/melfilter48/train/"
testFilesExtension = '.mfcc'
clusterFile = './data/k_mean_clusters.dat'

################    Loading cluster file    #######################
centroids = np.loadtxt(clusterFile)
n_clusters = centroids.shape[0]
n_input = centroids.shape[1] # input dimensionality ----> 585

################   Deleting existing files   ####################
try:
	os.remove('./data/train_melfilter48.dat')
except OSError:
	pass

try:
	os.remove('./data/train_mel48.dat')
except OSError:
	pass

################    Creating pair wise frames    #######################

print("Computing....")
print("Generating pair wise bag of code ....")
f_handle = open('./data/train_mel48.dat', 'ab')
if len(sys.argv) == 1:
	for root, dirs, files in os.walk(relativePathForTrain, topdown=False):
		for k in range(n_classes):
			for j in range(n_classes):
				if j > k: 
					for name in dirs:
						if name == dirs[k]:
							parts = []
							parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]
							for part in parts:
								example = np.loadtxt(os.path.join(root,name,part))
								i = 0
								rows, cols = example.shape
								context1 = np.zeros((rows-14,15*cols)) # 15 contextual frames
								while i <= (rows - 15):
									ex = example[i:i+15,:].ravel()
									ex = np.reshape(ex,(1,ex.shape[0]))
									context1[i:i+1,:] = ex
									i += 1
								r, c = context1.shape 
								actual_class_temp1 = name.split('c')
								actual_class_temp1.pop(0)
								actual_class1 = []
								for i in actual_class_temp1:
									actual_class1.append(int(i))


								for name2 in dirs:
									if name2 == dirs[j]:
										parts2 = []
										parts2 += [each for each in os.listdir(os.path.join(root,name2)) if each.endswith(testFilesExtension)]
										for part2 in parts2:
											example = np.loadtxt(os.path.join(root,name2,part2))
											i = 0
											rows, cols = example.shape
											context2 = np.zeros((rows-14,15*cols)) # 15 contextual frames
											while i <= (rows - 15):
												ex = example[i:i+15,:].ravel()
												ex = np.reshape(ex,(1,ex.shape[0]))
												context2[i:i+1,:] = ex
												i += 1
											r, c = context2.shape 			# rows and columns of context 

											context = np.concatenate((context1, context2), axis=0)
											bagOfcode = np.zeros(n_clusters)
											C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
											r, c = context.shape
											for i in range(r):
												bagOfcode[C[i]] += 1
											bag = bagOfcode
											actual_class_temp2 = name2.split('c')
											actual_class_temp2.pop(0)
											actual_class2 = []
											for i in actual_class_temp2:
												actual_class2.append(int(i))
											bag = np.append(bag,actual_class1[0])
											bag = np.append(bag,actual_class2[0])
											bag = np.reshape(bag,(1,bag.shape[0]))
											np.savetxt(f_handle, bag)
f_handle.close()

A = np.loadtxt('./data/train_mel48.dat')
np.random.shuffle(A)
np.savetxt('./data/train_melfilter48.dat',A)

