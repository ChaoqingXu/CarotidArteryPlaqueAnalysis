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

    ## remove the first row and column of "htmlplotcsv.csv" file and transpose it
    HTMLplot_list = HTMLplot_list.transpose()

    subject_Category_List = []
    for item in HTMLplot_list[:, 0]:
        if item == "subject_label": pass
        else:
            subject_Category_List.append(item)

    HTMLplot_list = np.delete(HTMLplot_list, 0, axis=1)
    HTMLplot_list = np.delete(HTMLplot_list, 0, axis=0)
    label_dictionary.pop("subject_label")
    
    return HTMLplot_list, label_dictionary, subject_Category_List



HTMLplot_list, label_dictionary, subject_Category_List = readHTMLCSVfile(HTMLCSVfile)
nsample, dim = np.shape(HTMLplot_list)
plaque_Assignment = mat(zeros((nsample, 2)))
y_plaque = []

for i in range(len(subject_Category_List)):
    if subject_Category_List[i].endswith('Calcium'):
        plaque_Assignment[i,:] = 0, 1
        y_plaque.append(0)
    if subject_Category_List[i].endswith('Fibrous'):
        plaque_Assignment[i, :] = 1, 1
        y_plaque.append(1)
    if subject_Category_List[i].endswith('IPH_lipid'):
        plaque_Assignment[i, :] = 2, 1
        y_plaque.append(2)
    if subject_Category_List[i].endswith('IPH'):
        plaque_Assignment[i, :] = 3, 1
        y_plaque.append(3)

print(plaque_Assignment) 
print(y_plaque)


# # shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
clf = OneVsRestClassifier( LinearSVC(random_state=0, dual=True, max_iter=100000) )

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
