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
from TobaccoData import *
from globvar import *
from time import process_time
from runClustering import showCluster

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




def showMLclusters(X, p, n_classes):

    useScatterPlot = 0

    numSamples = len(p)

    MLclusterAssignment = mat(zeros((numSamples, 2)))
    for i in range( numSamples):
        MLclusterAssignment[i, :] = float(p[i]), 1

    print("MLclusterAssignment = " + str(MLclusterAssignment))

    acc_score = precision_recall_fscore_support(y, p)
    # use f1 instead of acc
    acc = (acc_score[2][0] + acc_score[2][1] + acc_score[2][2]) / 3

    print("acc_score = " + str( acc_score[2] ))
    print("acc = " + str( acc ) )

    ns, nx = X.shape
    d2_train_dataset = X.reshape((ns, nx))
    pca = PCA(n_components=2)
    X = pca.fit(d2_train_dataset).transform(d2_train_dataset)

    numSamples, dim = X.shape

    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['royalblue', 'firebrick', 'darkgrey', 'forestgreen', 'teal', 'slategrey',
            'goldenrod', 'darkorange', 'tomato', 'mediumpurple', 'palevioletred']

    if n_classes > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

    if useScatterPlot: 
        color = []
        for i in range(numSamples):
            markIndex = int(MLclusterAssignment[i, 0])
            color.append( mark[markIndex] )
        scatter = plt.scatter(X[:, 0], X[:, 1], c=color, alpha=0.6, s=80, label=color)

        for i in range(numSamples):
            sampleID = str(i)
            plt.annotate(sampleID,  # this is the text
                            # these are the coordinates to position the label
                            (X[i, 0], X[i, 1]-0.016),
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 0),  # distance from text to points (x,y)
                            fontsize=8,
                            ha='center')  # horizontal alignment can be left, right or center
        plt.text(1.0, -1.30, 'ACC =' + str(("%.3f" % acc)), style='normal')

        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the centroids
        plt.title("Aroma" + '_' + "OnevsAll" +
                '_' + "k." + str(n_classes))
        plt.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        plt.savefig(destPath + "/" + "Aroma" + '_' +
                    "OnevsAll" + '_' + "Accuracy"  + '.png')
        plt.show()
        
    if not useScatterPlot:

        AromaList = list(AromaToLabel)
        # draw all samples
        for i in range(numSamples):
            markIndex = int(MLclusterAssignment[i, 0])
            marker_style = dict(
                color=mark[markIndex], marker='o', markerfacecoloralt='tab:red', markersize=11)
            plt.plot(X[i, 0], X[i, 1],
                     fillstyle='full', **marker_style, alpha=0.6)

            # label = "{:.2f}".format(y
            label = str(i)
            plt.annotate(label,  # this is the text
                         # these are the coordinates to position the label
                         (X[i, 0], X[i, 1]-0.016),
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 0),  # distance from text to points (x,y)
                         fontsize=8,
                         ha='center')  # horizontal alignment can be left, right or center

        marker_style_Fresh = dict(
            color=mark[0], marker='o', markerfacecoloralt='tab:red', markersize=9)
        marker_style_Neutral = dict(
            color=mark[1], marker='o', markerfacecoloralt='tab:red', markersize=9)
        marker_style_Brunt = dict(
            color=mark[2], marker='o', markerfacecoloralt='tab:red', markersize=9)

        plt.plot(1.0, -0.85, fillstyle='full', **
                 marker_style_Fresh, alpha=0.6,)
        plt.plot(1.0, -1.00, fillstyle='full', **marker_style_Neutral, alpha=0.6)
        plt.plot(1.0, -1.15, fillstyle='full', **marker_style_Brunt, alpha=0.6)

        plt.text(1.2, -0.85, str(AromaList[0]) + " =" + str(
            ("%.3f" % acc_score[2][0])), style='normal', fontsize="small")
        plt.text(1.2, -1.00, str(AromaList[1]) + " =" + str(
            ("%.3f" % acc_score[2][1])), style='normal', fontsize="small")
        plt.text(1.2, -1.15, str(AromaList[2]) + " =" + str(
            ("%.3f" % acc_score[2][2])), style='normal', fontsize="small")
        plt.text(1.2, -1.30, 'ACC =' + str(("%.3f" % acc)), style='normal', fontsize="small")

        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # draw the centroids
        plt.title("Aroma" + '_' + "OnevsAll" +
                '_' + "k." + str(n_classes))
        plt.savefig(destPath + "/" + "Aroma" + '_' +
                    "OnevsAll" + '_' + "Accuracy"  + '.png')
        plt.show()
    


def plotROCcurve(y_test, y_score):

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="deeppink",
        linestyle="-",
        linewidth=2,
    )

    AromaList = list(AromaToLabel)
    colors = cycle(['royalblue', 'firebrick', 'darkgrey'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            linewidth=2,
            linestyle=":",
            label="ROC curve of " + str(AromaList[i]) + " (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aroma ROC curve")
    plt.legend(loc="lower right", prop={'size': 9})
    plt.savefig(destPath + "/" + "Aroma" + '_' +
                "OnevsAll" + '_' + "ROC_Curve" + '.png')
    plt.show()



X, regionClusterAssignment, regionClusters, partClusterAssignment, partClusters, aromaClusterAssignment, aromaClusters = loadFileListFeatures(fileList)

y = []
aromaList = np.array( aromaClusterAssignment[:, 0] )
for i in range( len( aromaList )):
    tmp = aromaList[ i ][0]
    y.append( tmp )

X = normalize(X, axis=0, norm='max')

print("X = " + str( np.shape(X) ))
print( "X[0] = " + str(X[0]))
print("y = " + str(np.shape(y)) )
print( y )


# #-------------------------------------------------------------------#
#https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
#It might even indicate that you have some in appropriate features or strong correlations in the features. Debug those first before taking this easy way out.

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
clf = OneVsRestClassifier( LinearSVC(random_state=0, dual=True, max_iter=100000) )

# show clusters under DR
n_classes = len(AromaToLabel)
p = clf.fit(X_train, y_train).predict(X)
print("p = " + str( p ) )
print ("len(p) = " + str(len(p ) ) )
showMLclusters(X, p, n_classes)


# #-------------------------------------------------------------------#
# Binarize the output

y = label_binarize(y, classes=[0, 1, 2])
n_classes = 3

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)

# classifier
clf = OneVsRestClassifier(
    LinearSVC(random_state=0, dual=True, max_iter=100000))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

print( "yscore = " + str(np.shape(y_score)) + "  " + str(y_score[0]) )
print("y_test = " + str(np.shape(y_test)) + "  " + str(y_test[0]))

plotROCcurve(y_test, y_score)
