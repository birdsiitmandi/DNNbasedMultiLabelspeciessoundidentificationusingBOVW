 #!/usr/local/bin/python

#########################################################################
# K-Mean clustering on all training examples                            #
# Author: Ashish Arya                                                   #
# Date Created: 14/10/2017                                              #
# Purpose: 1) To perform K-means clustering on aggregate data           #
#                                                                       #
#                                                                       #
#########################################################################

import numpy as np
import pickle
import os
import warnings
from sklearn.cluster import KMeans

path = "./data/"
print("Computing clusters...")
train = np.loadtxt(path + 'k_mean_train_melfilter48.dat')

################   Deleting existing files   ####################
try:
	os.remove('./data/k_mean_clusters.dat')
except OSError:
	pass

#####################     K-Means clustering     #################

def kMeansClustering(X, K, maxIters = 100):
    result = KMeans(n_clusters=K, max_iter=maxIters, random_state=None, copy_x=True, n_jobs=1).fit(X)
    return result.cluster_centers_

no_clusters = 256          # Number of clusters
centroids = kMeansClustering(train,no_clusters)
np.savetxt('./data/k_mean_clusters.dat',centroids)
