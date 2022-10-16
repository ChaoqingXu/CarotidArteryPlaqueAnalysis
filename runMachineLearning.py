import sys
import os
import csv
import ast
from pathlib import Path
import platform
import numpy as np
import subprocess
import time
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from time import process_time
from vectors import Point, Vector

from numpy import *
from time import process_time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.decomposition import PCA

from radiomicsData import*


if platform.system() == 'Windows':
    sys.path.append(
        "D:\OverleafProj\CarotidArteryPlaqueAnalysis")

dirpath = os.getcwd()
csvHTMLPath = dirpath + "/CarotidArteryData_HTML"

HTMLCSVfile = csvHTMLPath + "/HTMLCSVfile.csv"

def readHTMLCSVfile(HTMLCSVfile):
    ## read integrated CSV file "HTMLCSVfile.csv", then transpose and delete unuseful collumns.
    HTMLplot_list = []
    with open(HTMLCSVfile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        g = []
        for row in reader:
            g.append(row)
        g = np.array(g)
        g = np.delete(g, np.s_[1:36], axis=1)  # remove column 1-35
        g = np.transpose(g)
        HTMLplot_list = g
        
    ## replace the first column with index
    subj_label_index_list = []
    subj_label_list = HTMLplot_list[:, 0]
    for i in range(len(subj_label_list)):
        if i == 0:
            subj_label_index_list.append(subj_label_list[i])
        else:
            subj_label_index_list.append(i)

    ## dictionary: featureLabel and index
    label_keys = HTMLplot_list[:, 0]
    label_index = subj_label_index_list
    label_dictionary = dict(zip(label_keys, label_index))

    HTMLplot_list[:, 0] = subj_label_index_list

    ## get submatrix from the HTMLplot_list, and normalize it
    row_num = len(HTMLplot_list)
    column_num = len(HTMLplot_list[0])
    tmpplot_list = np.delete(HTMLplot_list, 0, axis=1)
    tmpplot_list = np.delete(tmpplot_list, 0, axis=0)
    # normalize matrix by row
    tmpplot_normed = normalize(tmpplot_list, axis=1, norm='l1')
    HTMLplot_list[1:row_num, 1:column_num] = tmpplot_normed

    return HTMLplot_list, label_dictionary


HTMLplot_list, label_dictionary = readHTMLCSVfile(HTMLCSVfile)
HTMLplot_list = HTMLplot_list.transpose()
numSamples, dim = HTMLplot_list.shape

print(numSamples)
print(dim)

#    aromaClusterAssignment = mat(zeros((numSamples, 2)))

#    for i in range(numSamples):
#        if regionList[i] in RegionAbbreviationToLabel.keys():
#             regionClusterAssignment[i, :] = float(
#                RegionAbbreviationToLabel.get(regionList[i]) - 1), 1
#         if str(partList[i]) in PartToLabel.keys():
#             partClusterAssignment[i, :] = float(PartToLabel.get(partList[i]) - 1), 1
#         if aromaList[i] in AromaToLabel.keys():
#             aromaClusterAssignment[i, :] =  float(AromaToLabel.get(aromaList[i]) - 1), 1

# X, regionClusterAssignment, regionClusters, partClusterAssignment, partClusters, aromaClusterAssignment, aromaClusters = loadFileListFeatures(
#     fileList)

# y = []
# aromaList = np.array( aromaClusterAssignment[:, 0] )
# for i in range( len( aromaList )):
#     tmp = aromaList[ i ][0]
#     y.append( tmp )

# X = normalize(X, axis=0, norm='max')

# print("X = " + str( np.shape(X) ))
# print( "X[0] = " + str(X[0]))
# print("y = " + str(np.shape(y)) )
# print( y )


# # #-------------------------------------------------------------------#
# #https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
# #It might even indicate that you have some in appropriate features or strong correlations in the features. Debug those first before taking this easy way out.

# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# clf = OneVsRestClassifier( LinearSVC(random_state=0, dual=True, max_iter=100000) )

# # show clusters under DR
# n_classes = len(AromaToLabel)
# p = clf.fit(X_train, y_train).predict(X)
# print("p = " + str( p ) )
# print ("len(p) = " + str(len(p ) ) )
# showMLclusters(X, p, n_classes)


# # #-------------------------------------------------------------------#
# # Binarize the output

# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = 3

# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=0)

# # classifier
# clf = OneVsRestClassifier(
#     LinearSVC(random_state=0, dual=True, max_iter=100000))
# y_score = clf.fit(X_train, y_train).decision_function(X_test)

# print( "yscore = " + str(np.shape(y_score)) + "  " + str(y_score[0]) )
# print("y_test = " + str(np.shape(y_test)) + "  " + str(y_test[0]))

# plotROCcurve(y_test, y_score)
