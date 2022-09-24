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


# csvPath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis\\CarotidArteryData'
csvPath = r'D:\\OverleafProj\\CarotidArteryPlaqueAnalysis'

dataRow = [ 'Image type',\
            'Feature Class',\
            'Feature Name',\
            'Segmentation_segment_artery',\
            'Segmentation_segment_capillary',\
            'Segmentation_segment_fat',\
            'Segmentation_segment_tissue']

# read radiomics csv data and reorganize it as a standard format 
def readData(csvPath):
    csvFiles = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]
    for f in csvFiles: 
        if f == '91307196_L.csv':  
            with open(csvPath +"\\" + f, 'r') as csvfile:
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
                segmentArteryList, segmentCapillaryList, segmentFatList, segmentTissueList = [], [], [], [], [], [], []
                for i in range(len(gRow)):
                    if gRow[i] == 'Image type':
                        imageTypeList = g[:, [i]]
                    if gRow[i] == 'Feature Class':
                        featureClassList = g[:, [i]]
                    if gRow[i] == 'Feature Name':
                        featureNameList = g[:, [i]]
                    if gRow[i] == 'Segmentation_segment_artery':
                        segmentArteryList = g[:, [i]]
                    if gRow[i] == 'Segmentation_segment_capillary':
                        segmentCapillaryList = g[:, [i]]
                    if gRow[i] == 'Segmentation_segment_fat':
                        segmentFatList = g[:, [i]]
                    if gRow[i] == 'Segmentation_segment_tissue':
                        segmentTissueList = g[:, [i]]
                gFormat = np.hstack((imageTypeList, featureClassList, featureNameList,
                           segmentArteryList, segmentCapillaryList, segmentFatList, segmentTissueList))
           
                ## convert to split 4 labels and featureID
                imageTypeList_tmp, featureClassList_tmp, featureNameList_tmp = [], [], []
                segmentArteryList_tmp, segmentCapillaryList_tmp, segmentFatList_tmp, segmentTissueList_tmp = [], [], [], []
                for a, b, c, d, e, f, g in zip(imageTypeList, featureClassList, featureNameList, segmentArteryList, segmentCapillaryList, segmentFatList, segmentTissueList):
                    imageTypeList_tmp.append(a[0])
                    featureClassList_tmp.append(b[0]) 
                    featureNameList_tmp.append(c[0])
                    segmentArteryList_tmp.append(d[0])
                    segmentCapillaryList_tmp.append(e[0])
                    segmentFatList_tmp.append(f[0])
                    segmentTissueList_tmp.append(g[0])

                featureID = ({
                    'Image type': imageTypeList_tmp,
                    'Feature Class': featureClassList_tmp,
                    'Feature Name': featureNameList_tmp
                })
                df = pd.DataFrame(featureID)
                df["featureID"] = df['Image type'] + "_" + df['Feature Class'] + "_" + df['Feature Name']
                featureID = df["featureID"].to_numpy()
                segmentArteryList, segmentCapillaryList, segmentFatList, segmentTissueList = segmentArteryList_tmp, segmentCapillaryList_tmp, segmentFatList_tmp, segmentTissueList_tmp

                gFormat_FeautreID = np.vstack((featureID, segmentArteryList, segmentCapillaryList, segmentFatList, segmentTissueList))
                gFormat_FeautreID = gFormat_FeautreID.transpose()

                print(gFormat)
                print(np.shape(gFormat))
                print("-----------------------")
                print(gFormat_FeautreID)
                print(np.shape(gFormat_FeautreID))

                return gFormat, gFormat_FeautreID

readData(csvPath)
