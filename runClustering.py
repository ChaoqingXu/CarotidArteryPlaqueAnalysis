# -*- coding: utf-8 -*-

import sys
import os
import ast 
from pathlib import Path
import platform
import numpy as np
import subprocess
from time import process_time
from vectors import Point, Vector
import math

from numpy import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize

from fastdtw import fastdtw
import multiprocessing
from multiprocessing import Pool
from functools import partial 
from time import process_time

from sklearnCluster import *
from runMachineLearning import *


if platform.system() == 'Windows':
    sys.path.append(
        "D:\OverleafProj\CarotidArteryPlaqueAnalysis")

dirpath = os.getcwd()
outputPath = dirpath + "/Output"



def euclDistance(vector1, vector2):
    return sqrt(sum(pow(vector2 - vector1, 2)))

def npEuclDistance( vector1, vector2 ):
    return np.linalg.norm( np.array( vector1 ) - np.array( vector2 ) )


def eDistance( dataSet, dim, i , j ):
    sumSquares = 0 
    for k in range( dim ):
        sumSquares += ( dataSet[ i, k ] - dataSet[ j, k ] )**2
    return math.sqrt( sumSquares )


def distancePointToClosestCenter( dataSet, numSamples, dim, x, centroidsIndex ):
    result = eDistance( dataSet, dim, x, centroidsIndex[ 0 ] )
    for centroid in centroidsIndex[ 1: ]:
        distance = eDistance( dataSet, dim, x, centroid )
        if distance < result:
            result = distance
    return result


def initCentroids( dataSet, k ):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroidsIndex = []
    total = 0
    ## random select a centroid point
    firstCenter = random.choice( range( numSamples ) )
    centroidsIndex.append( firstCenter )

    for i in range( 0, k-1 ):
        weights = [ distancePointToClosestCenter( dataSet, numSamples, dim, x, centroidsIndex ) for x in range( numSamples ) ]
        total = sum( weights )
        # normalization
        weights = [ x/total for x in weights ]

        num = random.random()
        total = 0
        x = -1
        while total < num: 
            x+=1
            total += weights[ x ]
        centroidsIndex.append( x )

    for i in range( len( centroidsIndex ) ):        
        centroids[i, :] = dataSet[ centroidsIndex[ i ], :]

    return centroidsIndex, centroids



# k-means cluster
def kmeans(dataSet, k ):
    
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    # print( "clusterAssment init = " + str( clusterAssment ) )
    print( "---------------------------------------" )


    ## step 1: init centroids
    centroidsIndex, centroids = initCentroids( dataSet, k )

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

                ## step 4: update centroids
        for j in range(k):
            # clusterAssment[:,0].A==j???????????????clusterAssment???????????????????????????j???????????????????????????????????????array?????????????????????array?????????j?????????
            # ???dataSet???????????????????????????????????????
            # ???????????????j???????????????????????????
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print('Congratulations, cluster complete!')

    # print( "clusterAssment = " + str( clusterAssment ) )
    print( "---------------------------------------" )


    # print ( "centroids = " + str( centroids ) )
    # print ( "clusterAssment = " + str( clusterAssment.tolist() ) )

    # centroids???k????????????????????????????????????????????????
    # clusterAssment????????????????????????????????????????????????????????????????????????????????????????????????

    return centroidsIndex, centroids, clusterAssment
    
####################################################################################

# show your cluster only available with 2-D data
def showCluster(destPath, inputFileName, dataSet, clusterMethod, clusterNum, clusterAssment):

    ns, nd = dataSet.shape
    X = []
    for i in range(ns):
        tmp = []
        for j in range(nd):
            tmp.append(np.float64(HTMLplot_list[i][j]))
        X.append(tmp)

    d2_train_dataset = np.array(X).reshape((ns,nd))
    pca = PCA( n_components=2 )
    dataSet = pca.fit(d2_train_dataset).transform(d2_train_dataset)
    numSamples, dim = dataSet.shape

    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # mark = ['#940404', '#adacac', '#3570b5']
    mark = ['royalblue', 'firebrick', 'darkgrey','forestgreen', 'teal', 'slategrey', 'goldenrod', 'darkorange', 'tomato', 'mediumpurple', 'palevioletred']

    if clusterNum > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        # plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex]) 
        marker_style = dict(
            color=mark[markIndex], marker='o', markerfacecoloralt='tab:red', markersize=11)
        plt.plot(dataSet[i, 0], dataSet[i, 1], fillstyle='full', **marker_style, alpha=0.6)

        # label = "{:.2f}".format(y
        label = str( i )
        plt.annotate(label,  # this is the text
                     # these are the coordinates to position the label
                     (dataSet[i, 0], dataSet[i, 1]-0.016),
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 0),  # distance from text to points (x,y)
                    fontsize = 8,
                    ha='center')  # horizontal alignment can be left, right or center

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    # for i in range(clusterNum):
    #     plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12, c = 'black' ) 
    plt.title(inputFileName + '_' + clusterMethod + '_' + "k." + str(clusterNum))
    plt.savefig(destPath + "/" + inputFileName + '_' + clusterMethod + '_' + "k." + str(clusterNum) + '.png')
    plt.show()


def CarotidArteryClustering(destPath, dataSet, inputFileName, clusterMethod, clusteringResultFile, kValue):
  
    print("step 2: clustering...")
    dataSet = mat(dataSet)

    if clusterMethod == "kmeans":
        centroidsIndex, centroids, clusterAssment  = useKmeansPlusPlusClustering( dataSet, kValue )
    if clusterMethod == "Agglomerative":
        centroidsIndex, centroids, clusterAssment = useAgglomerativeClustering(dataSet, kValue)
    if clusterMethod == "Spectral":
        centroidsIndex, centroids, clusterAssment = useAgglomerativeClustering(dataSet, kValue)
    if clusterMethod == "OPTICS":
        centroidsIndex, centroids, clusterAssment = useAgglomerativeClustering(dataSet, kValue)
    if clusterMethod == "DBSCAN":
        centroidsIndex, centroids, clusterAssment = useAgglomerativeClustering(dataSet, kValue)


    # print( "write to a clusterAssment file " )
    dataAssign = clusterAssment.tolist()
    with open(destPath + "/" + inputFileName + "_" + clusteringResultFile + "_k." + str(kValue) + ".txt", 'w'):
        pass  # clean the file
    with open(destPath + "/" + inputFileName + "_" + clusteringResultFile + "_k." + str(kValue) + ".txt", "w") as f:
        for result in dataAssign[0:-1]: 
            f.write(str(result[0]) + ' ')
        f.write(str(dataAssign[-1][0]))
    
    ###show clustering
    # print("step 3: show the result...")
    showCluster(destPath, inputFileName, dataSet, clusterMethod, kValue, clusterAssment)
    ## ----------------------------------------------------##



        
# clusterMethod = int( sys.argv[1] )
# kValue = int( sys.argv[2] )

if __name__ == '__main__':
   
    print("start clustering")

    clusterMethod = "kmeans"
    kValue = 3

    HTMLplot_list, label_dictionary, subject_Category_List = readHTMLCSVfile(HTMLCSVfile)

    if clusterMethod == "kmeans":
       clusteringResultFile = "KmeansClusteringResult"
    if clusterMethod == "Agglomerative":
       clusteringResultFile = "AgglomerativeClusteringResult"
    if clusterMethod == "Spectral":
       clusteringResultFile = "SpectralClusteringResult"
    if clusterMethod == "OPTICS":
       clusteringResultFile = "OPTICSClusteringResult"
    if clusterMethod == "DBSCAN":
       clusteringResultFile = "DBSCANClusteringResult"

    CarotidArteryClustering(outputPath, HTMLplot_list, "CarotidArtery",
                            clusterMethod, clusteringResultFile, kValue)




