# -*- coding: utf-8 -*-

from sklearn.preprocessing import normalize
import glob
import shutil
import os
import math
import codecs
import numpy as np
from math import sqrt, acos, pi
import pandas as pd


from os import listdir
from os.path import isfile, join
import csv


csvPath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis\\CarotidArteryData'
csvSplitPath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis\\CarotidArteryData_SpilitLabels'
csvMergePath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis\\CarotidArteryData_MergeCSV'
csvHTMLPath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis\\CarotidArteryData_HTML'


dataRow = [ 'Image type',\
            'Feature Class',\
            'Feature Name',\
            'Segmentation_segment_Calcium',\
            'Segmentation_segment_Fibrous',\
            'Segmentation_segment_IPH',\
            'Segmentation_segment_IPH_lipid']


## define function for writing to spilit csv files 
def writeDataToSplitCSV(csvfile, featureID, segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList):

    print("write " + str(csvfile))
    fileName = os.path.basename(csvfile).split('.')[0]

    csvList = ['featureID', 'segment_Calcium',
               'segment_Fibrous', 'segment_IPH_lipid', 'segment_IPH']
    labelList = [featureID, segmentCalciumList,segmentFibrousList, segmentIPHlipidList, segmentIPHList]

    for i in range( len(csvList) ):
        with open(csvSplitPath + "\\" + fileName + "_" + csvList[i] + ".csv", 'w'):
            pass  # clean the file
        with open(csvSplitPath +"\\" + fileName + "_" + csvList[i] + ".csv", 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(labelList[i])
        print("write " + fileName + " csv ")



# read radiomics csv data, reorganize it as a standard format(correct the feature order), and write to split csv files
def SpilitSourceCSV(csvPath):
    csvFiles = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]

    for fileName in csvFiles: 
        with open(csvPath + "\\" + fileName, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            g = []
            for row in reader:
                g.append( row )
            g = np.array(g)

            ## reorganize the csv file
            tmp = g[[0], :] #???0???
            gRow = []
            for i in range(len( tmp[0])):
                gRow.append(tmp[0][i])
            imageTypeList, featureClassList, featureNameList,\
            segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList = [], [], [], [], [], [], []
            for i in range(len(gRow)):
                if gRow[i] == 'Image type': 
                    imageTypeList = g[:, [i]]
                if gRow[i] == 'Feature Class':
                    featureClassList = g[:, [i]]
                if gRow[i] == 'Feature Name':
                    featureNameList = g[:, [i]]
                if gRow[i] == 'Segmentation_segment_Calcium':
                    segmentCalciumList = g[:, [i]]
                if gRow[i] == 'Segmentation_segment_Fibrous':
                    segmentFibrousList = g[:, [i]]
                if gRow[i] == 'Segmentation_segment_IPH_lipid':
                    segmentIPHlipidList = g[:, [i]]
                if gRow[i] == 'Segmentation_segment_IPH':
                    segmentIPHList = g[:, [i]]

            # gFormat = np.hstack((imageTypeList, featureClassList, featureNameList,
            #             segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList))

            ## convert to split 4 labels and featureID

            imageTypeList_tmp, featureClassList_tmp, featureNameList_tmp = [], [], []
            segmentCalciumList_tmp, segmentFibrousList_tmp, segmentIPHlipidList_tmp, segmentIPHList_tmp = [], [], [], []
            for a, b, c, d, e, f, g in zip(imageTypeList, featureClassList, featureNameList, segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList):
                imageTypeList_tmp.append(a[0])
                featureClassList_tmp.append(b[0]) 
                featureNameList_tmp.append(c[0])
                segmentCalciumList_tmp.append(d[0])
                segmentFibrousList_tmp.append(e[0])
                segmentIPHlipidList_tmp.append(f[0])
                segmentIPHList_tmp.append(g[0])

            featureID = ({
                'Image Type': imageTypeList_tmp,
                'Feature Class': featureClassList_tmp,
                'Feature Name': featureNameList_tmp
            })

            df = pd.DataFrame(featureID)
            df["featureID"] = df['Image Type'] + "_" + df['Feature Class'] + "_" + df['Feature Name']
            featureID = df["featureID"].to_numpy()

            segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList = segmentCalciumList_tmp, segmentFibrousList_tmp, segmentIPHlipidList_tmp, segmentIPHList_tmp

            gFormat_FeautreID = np.vstack((featureID, segmentCalciumList, segmentFibrousList, segmentIPHlipidList, segmentIPHList))
            gFormat_FeautreID = gFormat_FeautreID.transpose()

            writeDataToSplitCSV(fileName, featureID, segmentCalciumList,
                                segmentFibrousList, segmentIPHlipidList, segmentIPHList)


def writeMergefile(sourceFileList,  targetFile):
    with open(targetFile, 'w'):
        pass  # clean the file
    for i in sourceFileList:
        fr = open(i, 'r').read()
        with open(targetFile, 'a') as f:
            f.write(fr)


def MergeCSV(sourcePath,  outputPath):

    keys = ['featureID', 'Calcium', 'Fibrous', 'IPH_lipid', 'IPH']
    csvFiles = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]

    for key in keys:
        if key is 'featureID':
            targetFile = outputPath + "/" + key + ".csv"
            for fileName in csvFiles:
                if key in fileName:
                    sourceFile = sourcePath + "/" + fileName 
                    shutil.copy(sourceFile, targetFile)  ## copy and rename 
            
        if ( key is not 'featureID') and (key is not 'IPH'):
            targetFile = outputPath + "/" + key + ".csv"
            sourceFileList = []
            for fileName in csvFiles:
                if key in fileName:
                    sourceFile = sourcePath + "/" + fileName
                    sourceFileList.append( sourceFile )
            writeMergefile(sourceFileList,  targetFile)

        if key is 'IPH':
            targetFile = outputPath + "/" + key + ".csv"
            sourceFileList = []
            for fileName in csvFiles:
                if ('IPH' in fileName) and ('IPH_lipid' not in fileName):
                    sourceFile = sourcePath + "/" + fileName
                    sourceFileList.append(sourceFile)
            writeMergefile(sourceFileList,  targetFile)


def writeHTMLplotCSV(sourceFileList, targetFile):
    with open(targetFile, 'w'):
        pass  # clean the file
    for i in sourceFileList:
        filename = os.path.basename(i).split('.')[0]
        fr = open(i, 'r').read()
        with open(targetFile, 'a') as f:
            f.write( str(filename) +',' )
            f.write(fr)


## integrate CSV files to a CSV for HTML ploting 
def HTMLplotCSV(sourcePath,  outputPath):
    keys = ['featureID', 'Calcium', 'Fibrous', 'IPH_lipid', 'IPH']
    csvFiles = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]
    HTMLCSVfile = outputPath + "/HTMLCSVfile.csv"
    htmlplotcsv = outputPath + "/htmlplotcsv.csv"

    for key in keys:
        if key is 'featureID':
            targetFile = outputPath + "/" + key + "_plot.csv"
            featureID_list = []
            for fileName in csvFiles:
                if key in fileName:
                    sourceFile = sourcePath + "/" + fileName
                    with open(sourcePath + "\\" + fileName, 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        g = []
                        for row in reader:
                            g.append(row)
                        g = np.array(g)
                        tmp = g[[0], :]
            featureID_list = tmp[0] 
            featureID_list = np.insert(featureID_list, 0, "subject_label")
            
            with open(targetFile, 'w'):
                pass  # clean the file
            with open(targetFile, 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(featureID_list)

        if (key is not 'featureID') and (key is not 'IPH'):
            targetFile = outputPath + "/" + key + ".csv"
            sourceFileList = []
            for fileName in csvFiles:
                if key in fileName:
                    sourceFile = sourcePath + "/" + fileName
                    sourceFileList.append(sourceFile)
            writeHTMLplotCSV(sourceFileList, targetFile)
        
        if key is 'IPH':
            targetFile = outputPath + "/" + key + ".csv"
            sourceFileList = []
            for fileName in csvFiles:
                if ('IPH' in fileName) and ('IPH_lipid' not in fileName):
                    sourceFile = sourcePath + "/" + fileName
                    sourceFileList.append(sourceFile)
            writeHTMLplotCSV(sourceFileList,  targetFile)

        ## merge files to a csv 
        htmlCSV = [f for f in listdir(outputPath) if isfile(join(outputPath, f))]
        htmlCSVList = []
        for key in keys:
            if (key is not 'IPH'):
                for fileName in htmlCSV:
                    if key in fileName:
                        htmlCSVList.append(outputPath + "/" + fileName)
            if key is 'IPH':
                for fileName in htmlCSV:
                    if ('IPH' in fileName) and ('IPH_lipid' not in fileName):
                        htmlCSVList.append(outputPath + "/" + fileName)

        writeMergefile(htmlCSVList,  HTMLCSVfile)

    ## read integrated CSV file "HTMLCSVfile.csv", then transpose and delete unuseful collumns.
    HTMLplot_list = []
    with open(HTMLCSVfile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        g = []
        for row in reader:
            g.append(row)
        g = np.array(g)
        g = np.delete(g, np.s_[1:36], axis=1) ## remove column 1-35
        g = np.transpose(g)
        HTMLplot_list = g

    print(np.shape(HTMLplot_list))


    ## replace the first column with index
    subj_label_index_list = []
    subj_label_list = HTMLplot_list[:, 0]
    for i in range(len(subj_label_list) ):
        if i == 0:
            subj_label_index_list.append( subj_label_list[i] )
        else:
            subj_label_index_list.append( i )

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
    tmpplot_normed = normalize(tmpplot_list, axis=1, norm='l1')     # normalize matrix by row    
    HTMLplot_list[1:row_num, 1:column_num] = tmpplot_normed

    ## write to csv file
    with open(htmlplotcsv, 'w'):
        pass  # clean the file
    with open(htmlplotcsv, 'w', newline="") as file:
        writer = csv.writer(file)
        for subj_csv in HTMLplot_list:
            writer.writerow(subj_csv)

# SpilitSourceCSV(csvPath)    ## use for spliting csv file to label_csv files
# MergeCSV(csvSplitPath, csvMergePath)   ## use for machine learning 
# HTMLplotCSV(csvSplitPath, csvHTMLPath)  ## use for ploting 
