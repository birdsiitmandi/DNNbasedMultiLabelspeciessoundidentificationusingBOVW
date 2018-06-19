#!/usr/local/bin/python

#########################################################################
# 																		#
# Author: Deep Chakraborty												#
# Date Created: 18/05/2016												#
# Purpose: 1) To pickle the artificial data for use with tensorflow		#
# 		   2) To train a neural network with the loaded data 			#
# 		    		   													#
#########################################################################

import sys
import numpy as np
import tensorflow as tf
import os
import pickle
from scipy.stats import mode

################    All Constants and paths used    #####################
path = "./data/"
classes = np.loadtxt("classes.txt", dtype='str')
n_classes = classes.size
parametersFileDir = "./data/parameters_mfcc_2.pkl"
relativePathForTest = "./data/melfilter48/test_clear_pair/"
testFilesExtension = '.mfcc'
confMatFileDirectory = './data/confusion_multi.txt'
clusterFile = './data/k_mean_clusters.dat'
#classifiedTestDirectory = './testOutput/'

################    Data Loading and Plotting    ########################
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def indices(a, func):
	"""Finds elements of the matrix that correspond to a function"""
	return [i for (i, val) in enumerate(a) if func(val)]


#test_labels_dense = np.loadtxt('./data/ground_truth.txt');
#test_labels_dense = test_labels_dense.astype(int)
# test_y = dense_to_one_hot(test_labels_dense, num_classes = n_classes)
# print train_labels_dense
# plot_data(train_X, train_labels_dense)
# time.sleep(10)
# plt.close('all')
print("Data Loaded and processed ...")

################    Loading cluster file    #######################
centroids = np.loadtxt(clusterFile)
n_clusters = centroids.shape[0]
n_input = centroids.shape[1] # input dimensionality ----> 585
#print(n_clusters)
#print(n_input)

################## Neural Networks Training #################################

print("Verifying Neural Network Parameters ...")

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
#n_hidden_3 = 256 # 3rd layer num features
#n_hidden_4 = 256
#n_hidden_5 = 128
n_input = 16 # input dimensionality

x = tf.placeholder("float32", [None, n_input])
y = tf.placeholder("float32", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    #Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    #Hidden layer with sigmoid activation
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
	#layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
	#layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
	return tf.nn.sigmoid(tf.matmul(layer_2, _weights['out']) + _biases['out'])
	#return tf.nn.softmax(tf.matmul(layer_5, _weights['out']) + _biases['out'])

print("Loading saved Weights ...")
file_ID = parametersFileDir
f = open(file_ID, "rb")
W = pickle.load(f)
b = pickle.load(f)

# print "b1 = ", b['b1']
# print "b2 = ", b['b2']
# print "b3 = ", b['out']

weights = {
	'h1': tf.Variable(W['h1']),
	'h2': tf.Variable(W['h2']),
	#'h3': tf.Variable(W['h3']),
	#'h4': tf.Variable(W['h4']),
	#'h5': tf.Variable(W['h5']),
	'out': tf.Variable(W['out'])
	}

biases = {
	'b1': tf.Variable(b['b1']),
	'b2': tf.Variable(b['b2']),
	#'b3': tf.Variable(b['b3']),
	#'b4': tf.Variable(b['b4']),
	#'b5': tf.Variable(b['b5']),
	'out': tf.Variable(b['out'])
}
# print type(b['b1'])
# print type(biases['b1'])

f.close()

# layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))


pred = multilayer_perceptron(x, weights, biases)

print("Testing the Neural Network")
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	num_examples = 0
	for root, dirs, files in os.walk(relativePathForTest, topdown=False):
		for name in dirs:
			parts = []
			parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]

			for part in parts:
				num_examples += 1

	# Test model

	# likelihood = tf.argmax(tf.reduce_mean(pred, 0),1)
	test_labels_dense = np.zeros(num_examples)
	test_labels_dense = test_labels_dense.astype(int)
	label = np.zeros(test_labels_dense.shape[0])
	ind = 0
	gt = 0
	#print("Computing...")
	if len(sys.argv) == 1:
		for root, dirs, files in os.walk(relativePathForTest, topdown=False):
			list_actual_class = []
			list_predicted_class = []
			for name in dirs:
				#print(name)
				parts = []
				parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]

				for part in parts:
					#print("Part : ",part)

					example = np.loadtxt(os.path.join(root,name,part))
					i = 0
					rows, cols = example.shape
					context = np.zeros((rows-14,15*cols)) # 15 contextual frames
					while i <= (rows - 15):
						ex = example[i:i+15,:].ravel()
						ex = np.reshape(ex,(1,ex.shape[0]))
						context[i:i+1,:] = ex
						i += 1
					r, c = context.shape 			# rows and columns of context 
					bagOfcode = np.zeros(n_clusters)
					# Calculating the bag of code
					C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
					for i in range(r):
						bagOfcode[C[i]] += 1
					bag1 = bagOfcode
					bag1 = np.reshape(bag1,(1,bag1.shape[0]))
					see = tf.reduce_sum(pred,0)
					
					#Another way to predict class
					# product = np.argmax(np.asarray(see.eval({x: bag1})))
					# product = np.asarray(see.eval({x: bag1}))
					# print(product)
					# avg_classes = 2
					# predicted_class = product.argsort()[-avg_classes:][::-1]

					ptemp = str(part)
					p2temp = ptemp.split('.')
					product_count = (np.asarray(see.eval({x: bag1})))		#Contains a count of windows per class
					threshold = 0.9
					# print(threshold)
					predicted_class = []
					for j in range(n_classes):
						if(product_count[j] >= threshold):
							predicted_class.append(j)
					for i in range(len(predicted_class)):
						if predicted_class[i] == 0:
							predicted_class[i] = 26
					
					actual_class_temp = name.split('c')
					actual_class_temp.pop(0)
					actual_class = []
					for j in actual_class_temp:
						actual_class.append(int(j))
					#print("Actual : ",actual_class)
					#print("Predicted : ",predicted_class)
					print("\nActual Class : ")
					for k in (actual_class):
						print(classes[k-1])
					print("Predicted Class :")
					for k in (predicted_class):
						print(classes[k-1])	
					list_actual_class.append(actual_class)
					list_predicted_class.append(predicted_class)
					ind += 1
				gt += 1
		accuracy_multi = 0
		recall = 0
		precision = 0
		hamming_loss = 0
		for j in range(len(list_actual_class)):
			intersection = len(set(list_actual_class[j]) & set(list_predicted_class[j]))
			union = len(set(list_actual_class[j]) | set(list_predicted_class[j]))
			accuracy_multi += (intersection/union)
		accuracy_multi = (accuracy_multi/len(list_actual_class))*100
		print("~ Results ~")
		print("Accuracy is %.4f " % accuracy_multi)

	else:
		path = sys.argv[1]
		example = np.loadtxt(path)
		i = 0
		rows, cols = example.shape
		context = np.zeros((rows-14,15*cols)) # 15 contextual frames
		while i <= (rows - 15):
			ex = example[i:i+15,:].ravel()
			ex = np.reshape(ex,(1,ex.shape[0]))
			context[i:i+1,:] = ex
			i += 1
		r, c = context.shape 			# rows and columns of context 
		bagOfcode = np.zeros(n_clusters)
		# Calculating the bag of code
		C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in context])
		for i in range(r):
			bagOfcode[C[i]] += 1
		bag1 = bagOfcode
		bag1 = np.reshape(bag1,(1,bag1.shape[0]))
		see = tf.reduce_sum(pred,0)
		
		#Another way to predict class
		# product = np.argmax(np.asarray(see.eval({x: bag1})))
		# product = np.asarray(see.eval({x: bag1}))
		# print(product)
		# avg_classes = 2
		# predicted_class = product.argsort()[-avg_classes:][::-1]
		ptemp = str(part)
		p2temp = ptemp.split('.')
		product_count = (np.asarray(see.eval({x: bag1})))		#Contains a count of windows per class
		threshold = 0.9
		# print(threshold)
		predicted_class = []
		for j in range(n_classes):
			if(product_count[j] >= threshold):
				predicted_class.append(j)
		for i in range(len(predicted_class)):
			if predicted_class[i] == 0:
				predicted_class[i] = 26		
		#print("predicted classes", predicted_class)
		print("\nPridicted bird classes are : ")			
		for k in (predicted_class):
			print(classes[k-1])
		
