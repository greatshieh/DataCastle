'''
利用信息增益将连续特征离散化

'''
from math import log
import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer


# 计算香农熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for n in range(len(dataset)):
        currentLabel = dataset.iloc[n, -1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def discretization(df, label_name):
    df_Ent = {}
    baseEntropy = calcShannonEnt(df)
    print('基本信息增益是: %f'%baseEntropy)

    predictor = [x for x in df.columns if x != label_name]

    for x in predictor:
        bestInfoGain = 0.0
        df_Ent[x] = {}

        for row in range(len(df) - 1):
            newEntropy = 0.0
            sort_df = df[[x, label_name]].sort_values(by=x, ascending=True)
            if sort_df.iloc[row, 0] == sort_df.iloc[row + 1, 0]:
                continue
            split_point = (sort_df.iloc[row, 0] + sort_df.iloc[row + 1, 0]) / 2
            bin_encoder = Binarizer(split_point)
            sort_df[x] = bin_encoder.fit_transform(
                sort_df[x].values.reshape(-1, 1))
            for value in [0, 1]:
                subdataset = sort_df[sort_df[x] == value]
                prob = len(subdataset) / float(len(sort_df))
                newEntropy += prob * calcShannonEnt(subdataset)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                df_Ent[x]['best_point'] = split_point
                df_Ent[x]['Ent'] = infoGain
                bestInfoGain = infoGain

        print('%s的最佳划分点是 %f, 最大信息增益是 %f。' % (x, df_Ent[x]['best_point'],
                                             df_Ent[x]['Ent']))

    return df_Ent
