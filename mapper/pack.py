# this code is acquired from:
# https://blog.csdn.net/linweieran/article/details/100585052

import numpy as np
import math

# memory will exceed if use the number of weight element, use an offset to reduce memory usage
bias = 1000


# assigned layer is a list
# Gtime is a graph

def pack(AccObject, Gtime, AssignedLayers, bindedLayers):
    AccObject.getExtraWeightElement()
    C = math.ceil(AccObject.extraWeightElement / bias)

    AccObject.assiLayers = AssignedLayers
    AccObject.bindLayer = []

    # if it is modality switch, the previous binded layer should not be removed
    # add the binded layers to AccObject.bindLayer, reduce the available AccObject.extraWeightElement
    if bindedLayers:
        for layer in AssignedLayers:
            if layer in bindedLayers:
                AccObject.bindLayer.append(layer)
                bindLayerWei = math.ceil(Gtime.nodes[layer]['weight'] / bias)
                Gtime.nodes[layer]['bind'] = True
                C = C - bindLayerWei

        for element in AccObject.bindLayer:
            AssignedLayers.remove(element)


    Num = len(AssignedLayers)
    CapacityLeft = C
    WeightList = []
    WeightDic = {}
    TargetLayer = []
    for i in range(Num):
        layerWei = math.ceil(Gtime.nodes[AssignedLayers[i]]['weight'] / bias)
        WeightList.append(layerWei)
        WeightDic[AssignedLayers[i]] = layerWei

    # WeightDic.sort(reverse=True)
    WeightList.sort(reverse=True)
    # print(WeightList)
    WeightDic = sorted(WeightDic.items(), key=lambda item:item[1], reverse=True)
    for i in range(Num):
        # print(WeightDic[i][0])
        if CapacityLeft - WeightList[i] > 0:
            CapacityLeft = CapacityLeft - WeightList[i]
            # TargetLayer.append(WeightDic[WeightList[i]])
            TargetLayer.append(WeightDic[i][0])
        elif CapacityLeft - WeightList[i] < 0:
            continue
        elif CapacityLeft - WeightList[i] == 0:
            CapacityLeft = CapacityLeft - WeightList[i]
            # TargetLayer.append(WeightDic[WeightList[i]])
            TargetLayer.append(WeightDic[i][0])
            break

    for layer in TargetLayer:
        Gtime.nodes[layer]['bind'] = True
        AccObject.bindLayer.append(layer)

    return AccObject, Gtime







