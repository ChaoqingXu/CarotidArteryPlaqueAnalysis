# -*- coding: utf-8 -*-

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

    csvList = ['featureID','segmentArtery', 'segmentCapillary', 'segmentFat', 'segmentTissue']
    labelList = [featureID, segmentCalciumList,segmentFibrousList, segmentIPHlipidList, segmentIPHList]

    for i in range( len(csvList) ):
        with open(csvSplitPath +"\\" + fileName + "_" + csvList[i] + ".csv", 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(labelList[i])
        print("write " + fileName + " csv ")



# read radiomics csv data, reorganize it as a standard format(correct the feature order), and write to split csv files
def readData(csvPath):
    csvFiles = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]
    for fileName in csvFiles: 
        with open(csvPath + "\\" + fileName, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            g = []
            for row in reader:
                g.append( row )
            g = np.array(g)

            ## reorganize the csv file
            tmp = g[[0], :] #第0行
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





readData(csvPath)

# writeDataToSplitCSV('91307196_L',featureID, segmentCalciumList,
#                     segmentFibrousList, segmentIPHlipidList, segmentIPHList)
