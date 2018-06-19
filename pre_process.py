 #!/usr/local/bin/python

#########################################################################
# Aggregating all training data of all classes in one file				#
# Author: Ashish Arya  													#
# Date Created: 13/10/2017												#
# Purpose: 1) To aggregate the training data from all classes			#
# 		   																#
# 		    		   													#
#########################################################################

import numpy as np
import pickle
import os

################   Deleting existing files   ####################
try:
	os.remove('./data/k_mean_train_mel48.dat')
except OSError:
	pass

try:
	os.remove('./data/k_mean_train_melfilter48.dat')
except OSError:
	pass

###################    Aggregate data from all training examples of all classes    #################
sum = 0
f_handle = open('./data/k_mean_train_mel48.dat', 'ab')
for root, dirs, files in os.walk("./data/melfilter48/train", topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.mfcc')]
			print(name, "...")
			for part in parts:
				example = np.loadtxt(os.path.join(root,name,part))
				i = 0
				rows = example.shape[0]
				while i <= (rows - 15):
					ex = example[i:i+15,:].ravel()
					ex = np.reshape(ex,(1,ex.shape[0]))
					np.savetxt(f_handle, ex)
					sum += 1
					i += 1
				print("No. of context windows: %d" % i)
print("No. of Training examples: %d" % sum)
f_handle.close()

A = np.loadtxt('./data/k_mean_train_mel48.dat')
np.random.shuffle(A)
np.savetxt('./data/k_mean_train_melfilter48.dat',A,delimiter = ' ')
