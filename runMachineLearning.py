import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

outputPath = dirpath + "/Output"

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


def getTopKFeatureIndices(scores, nSelect):
    nf = min(nSelect, len(scores))
    return np.argsort(scores)[:nf]


def getExtraTreeScores(X, y, nFeat):
    # n_estimators = 150, min_samples_split = 4, max_features = nFeat, n_jobs = 4, random_state = int(round(time.time())) )
    forest = ExtraTreesClassifier(
        n_estimators=100, min_samples_split=4, max_features='sqrt', n_jobs=4)
    forest.fit(X, y)
    importances = forest.feature_importances_
    return -importances


def getRFScores(X, y, nFeat):
    forest = RandomForestClassifier(n_estimators=1500, min_samples_split=4,
                                    max_features=nFeat, n_jobs=4, random_state=int(round(time.time())))
    forest.fit(X, y)
    importances = forest.feature_importances_
    return -importances

    return pValues
    

def getLVMScores(X, y, nFeat):
    clfF = SVC(kernel="linear", C=0.2, probability=True,
               max_iter=1000000, tol=1e-10)
    clfF.fit(X, y)
    importances = np.abs(clfF.coef_[0])
    return -importances
    
def getFeatureScores( X, y, FeatureScoringMethod, nFeat ):
    if FeatureScoringMethod == "Extra Trees":
        return getExtraTreeScores( X, y, nFeat ) 
    elif FeatureScoringMethod == "Random Forest":
    	return getRFScores( X, y, nFeat )
    else:
    	return getLVMScores( X, y, nFeat ) #SVM

def plotROCcurve(y_test, y_score, n_classes, plaque_list, color_list, _path):

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
        label="macro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["macro"]),
        color="deeppink",
        linestyle="-",
        linewidth=2,
    )

    colors = cycle(color_list)
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            linewidth=2,
            linestyle=":",
            label="ROC curve of " +
            str(plaque_list[i]) + " (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Plaque ROC curve")
    plt.legend(loc="lower right", prop={'size': 9})
    plt.savefig(_path + "/" + "plaque" + '_' +
                "OnevsAll" + '_' + "ROC_Curve" + '.png')
    plt.show()


def multiClass_Classification(HTMLCSVfile, outputPath):
    HTMLplot_list, label_dictionary, subject_Category_List = readHTMLCSVfile(HTMLCSVfile)
    nsample, dim = np.shape(HTMLplot_list) #nsample = 92, dim = 389
    plaque_Assignment = mat(zeros((nsample, 2)))
    plaque_list = ['Calcium', 'Fibrous', 'IPH_lipid', 'IPH']
    color_list = ['royalblue', 'firebrick', 'darkgrey', 'green']

    X = []
    for i in range(nsample):
        tmp = []
        for j in range(dim):
            tmp.append(np.float64(HTMLplot_list[i][j]))
        X.append(tmp)

    y = []
    categoryList = []
    for i in range(len(subject_Category_List)):
        if subject_Category_List[i].endswith('Calcium'):
            plaque_Assignment[i, :] = 0, 1
            categoryList.append('Calcium')
            y.append(0)
        if subject_Category_List[i].endswith('Fibrous'):
            plaque_Assignment[i, :] = 1, 1
            categoryList.append('Fibrous')
            y.append(1)
        if subject_Category_List[i].endswith('IPH_lipid'):
            plaque_Assignment[i, :] = 2, 1
            categoryList.append('IPH_lipid')
            y.append(2)
        if subject_Category_List[i].endswith('IPH'):
            plaque_Assignment[i, :] = 3, 1
            categoryList.append('IPH')
            y.append(3)
    y_categoryList = y

    # # # shuffle and split training and test sets
    y = label_binarize(y, classes=[0, 1, 2, 3])
    n_classes = y.shape[1]

    useCrossValidation, useFeatureSelection, useRECF = 1, 1, 0
    numTrials = 3
    kfold = 5
    nFeat = 5
    featureRanks = np.zeros(shape=(kfold*numTrials, dim))
    
    if ( not useCrossValidation ) and ( not useFeatureSelection ):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = OneVsRestClassifier( LinearSVC(random_state=0, dual=True, max_iter=100000) )

        y_score = clf.fit(X_train, y_train).decision_function(X_test)

        plotROCcurve(y_test, y_score, n_classes, plaque_list, color_list, outputPath)
        print("finish the task")
    
    if (useCrossValidation) and (useFeatureSelection):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if(not useRECF):
            scores = getFeatureScores(X_train, y_train, "Extra Trees", nFeat)
            featureSubsetIndices = getTopKFeatureIndices(scores, nFeat)
            print(featureSubsetIndices) # top K feature indices
            ordered = np.argsort(scores)
            # print(ordered) # feature order
            X_train = X_train[:, featureSubsetIndices]
            X_test = X_test[:, featureSubsetIndices]

            topKfeaturenameList = []
            topKfeature = []
            X = np.array(X)
            for i in featureSubsetIndices:
                feature = list(label_dictionary.keys())[list(label_dictionary.values()).index(i + 1)]
                topKfeaturenameList.append( feature )  ## get feature name by top k features
                topKfeature.append(np.float64(X[:, i]))

            print(np.shape(topKfeature))
            print(np.shape(topKfeaturenameList))
            print(np.shape(categoryList))
            print(np.shape(y_categoryList))
            print(np.shape(plaque_list))

            topKfeature = np.transpose(topKfeature)
            # topKfeature = [topKfeature, categoryList]
            # topKfeaturenameList = topKfeaturenameList.append("category")

            # topKData = ({
            #     'topK feature List': topKfeaturenameList,
            #     'topK feature': topKfeature,
            #     'category List': categoryList
            # })
            

            df = pd.DataFrame(topKfeature, columns = topKfeaturenameList, index = y_categoryList)
            sns.pairplot(df)
            # sns.pairplot(df, hue="species", markers=["o", "s", "D"])
            plt.show()

        else:
            rfecv = RFECV(estimator=SVC, step=1,
                        cv=StratifiedKFold(3), scoring='accuracy')
            rfecv.fit(X_train, y_train)
            X_train = rfecv.transform(X_train)
            X_test = rfecv.transform(X_test)
            nfeat = rfecv.n_features_
            featureRanks[i, :] = rfecv.ranking_
            print("top features:  " + str(nfeat))

        clf = OneVsRestClassifier(
            LinearSVC(random_state=0, dual=True, max_iter=100000))

        y_score = clf.fit(X_train, y_train).decision_function(X_test)   


multiClass_Classification(HTMLCSVfile, outputPath)
## https://www.codespeedy.com/multiclass-classification-using-scikit-learn/
