import sys
import os
import numpy as np
import subprocess
from time import process_time

from numpy import *

from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans

# Python program to check if two  
# to get unique values from list 
# using traversal 
def unique(list1):  
    unique_list = [] 
    tmpList = []
    for i in range( len( list1 ) ):
        if not list1[ i ] in tmpList:
            unique_list.append( i )
            tmpList.append( list1[ i ] ) 
    return unique_list





## Kmeans++
def useKmeansPlusPlusClustering( dataSet, k ):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=20).fit(dataSet)
    labels = kmeans.labels_

    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    clusterAssment = mat(zeros((numSamples, 2)))

    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1

    centroidsIndex = unique( labels )
    for i in range( len( centroidsIndex ) ):        
        centroids[i, :] = dataSet[ centroidsIndex[ i ], :]
        
    return centroidsIndex, kmeans.cluster_centers_, clusterAssment




## hierachical clustering
def useAgglomerativeClustering( dataSet, k ):
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit( dataSet )
    labels = ward.labels_

    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    clusterAssment = mat(zeros((numSamples, 2)))

    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1
    
    centroidsIndex = unique( labels )
    for i in range( len( centroidsIndex ) ):        
        centroids[i, :] = dataSet[ centroidsIndex[ i ], :]
        
    return centroidsIndex, centroids, clusterAssment




def useAffinityPropagation( dataSet, k ):
    clustering = AffinityPropagation().fit( dataSet )
    AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=200, preference=None, verbose=False)
    
    labels = clustering.labels_

    numSamples, dim = dataSet.shape
    clusterAssment = mat(zeros((numSamples, 2)))
    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1

    return clustering.cluster_centers_indices_, clustering.cluster_centers_, clusterAssment


def userDBSCAN( dataSet, k ):
    clustering = DBSCAN(eps=3, min_samples=2).fit( dataSet )
    labels = clustering.labels_

    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    clusterAssment = mat(zeros((numSamples, 2)))

    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1
    
    # centroidsIndex = unique( labels )
    # for i in range( len( centroidsIndex ) ):        
    #     centroids[i, :] = dataSet[ centroidsIndex[ i ], :]
        
    return centroidsIndex, centroids, clusterAssment


def userOPTICS( dataSet, k ):
    clustering = OPTICS(eps=3, min_samples=2).fit( dataSet )
    labels = clustering.labels_

    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    clusterAssment = mat(zeros((numSamples, 2)))

    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1
    
    return centroidsIndex, centroids, clusterAssment



def useSpectralClustering( dataSet, k ):
    clustering = SpectralClustering(n_clusters= k, assign_labels="kmeans", gamma=1,
                random_state=0).fit( dataSet )
    print("clustering.labels_ = " + str(  clustering.labels_ ) )
    labels = clustering.labels_

    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    clusterAssment = mat(zeros((numSamples, 2)))

    for i in range( numSamples ):
        clusterAssment[i, :] = labels[ i ], 1
    
    centroidsIndex = unique( labels )
    for i in range( len( centroidsIndex ) ):        
        centroids[i, :] = dataSet[ centroidsIndex[ i ], :]
        
    return centroidsIndex, centroids, clusterAssment

