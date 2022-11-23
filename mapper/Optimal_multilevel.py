# -*- coding: utf-8 -*-
# author: Xinyi Zhang time:4/16/2021
import itertools
import sys
import pickle
import time
sys.path.append('C:\\Users\\walker\\Desktop\\\p2p_mapping\\FPGA2023\\FPGA2023\\AcceleratorV2')
sys.path.append('C:\\Users\\walker\\Desktop\\\p2p_mapping\\FPGA2023\\FPGA2023\\Modality')

from cnn_acc_1 import cnn_acc_1
from cnn_acc_2 import cnn_acc_2
from cnn_acc_3 import cnn_acc_3
from cnn_acc_4 import cnn_acc_4
# from cnn_acc_5 import cnn_acc_5
from cnn_acc_6 import cnn_acc_6
from cnn_acc_7 import cnn_acc_7
from cnn_acc_8 import cnn_acc_8
from cnn_acc_9 import cnn_acc_9
from fc_acc_ese_10 import fc_acc_ese
from fc_acc_goDeeper_11 import fc_acc_goDeeper
from lstm_acc_pynq_12 import lstm_acc_pynq
from lstm_acc_vc707_13 import lstm_acc_vc707
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import random
from Modality_case1 import Modality1
from Modality_case2 import Modality2
from Modality_case3 import Modality3
# from Modality_case4 import Modality4
from Modality_case5 import Modality5
from Modality_case6 import Modality6
from Modality_case7 import Modality7

import itertools
from pack import pack
import copy
from MapperBase import MapperBase

# the model input to AccMapper is a graph
class MapperInit(MapperBase):
    def __init__(self):
        super(MapperInit,self).__init__()
        self.Gmap = nx.DiGraph() # a graph that include layer dependency, and task dependency, will be empty after mapping
        self.Gmapstart = nx.DiGraph()   #是连续计算层的初始层的依赖关系
        self.Gtime = nx.DiGraph() # a graph to record layers status when mapping
        self.Gtimeaftermapping = nx.DiGraph()
        self.GtimeLastTune = nx.DiGraph()
        self.GtimeLastTuneTmp = nx.DiGraph()
        self.Gacc = nx.DiGraph() # a graph to record acc assigned layer hardware dependency after mapping
        self.GaccLastTune = nx.DiGraph()
        self.GaccLastTuneTmp = nx.DiGraph()
        self.modalities=[] # the different modalities
        self.AccList = [] # the accelerators (sent in as object)
        self.AccListLastTune = []  # the accelerators (sent in as object)
        self.AccListLastTuneTmp = []  # the accelerators (sent in as object)
        self.AccLayers = {} # the assigned layers on accs
        self.AccLayersLastTune = {}  # the assigned layers on accs
        self.Acctimeaftermapping = {} #此处是每个加速器在初步映射后的计算终止时间
        self.Acctimeafterbinding = {}
        self.Acctimeafterifmofm = {}
        self.AccLayersLastTuneTmp = {}
        # to store the raw mapping result of mapping, in each iteration of 'color flip'
        self.GtimeReserve = nx.DiGraph()
        self.GtimeReserveTmp = nx.DiGraph()
        self.GaccReserve = nx.DiGraph()
        self.GaccReserveTmp = nx.DiGraph()
        self.AccListReserve = []
        self.AccListReserveTmp = []
        self.AccLayersReserve = {}
        self.AccLayersReserveTmp = {}
        self.layers2Map = {}
        self.PlatformElapsed = float('inf')
        self.NetBand = 0
        self.Dtype = 0
        self.timeaftermapping = 0
        self.timeafterifmofm = 0
        self.energyaftermapping = 0
        self.energyafterbinding = 0
        self.energyafterifmofm = 0
        self.cost = {}
        self.costtmp = {}
        self.translataftermapping = {}
        self.complataftermapping = {}
        self.AccLatencymap = {}
        self.acclastlayer = {}
        self.acclastlayer = {}
        self.complatafterifmofm = 0
        self.AccListNoTouch = []
        self.GmapPermu = nx.DiGraph()  # a graph that include layer dependency, and task dependency, will be empty after mapping
        self.LayerOrder = []
        self.Permutations = []
        self.AccListPermu = []
        self.optimallat = []
        self.optimaltime = []
        self.h2plat = []
        self.h2ptime = []

    def getSysPower(self, timeGraph):
        allNodes = list(timeGraph.nodes)
        sysPower = 0
        for node in allNodes:
            accName = timeGraph.nodes[node]['acc']
            start = timeGraph.nodes[node]['start']
            end = timeGraph.nodes[node]['end']
            Acc = self.getAccbyName(accName, self.AccList)
            sysPower = sysPower + (end - start)* Acc.power
        return sysPower

        # get the acc list and init the acclayer assignment
    def getAccList(self, AccList):
        self.AccList = AccList
        self.AccListNoTouch = AccList
        self.getAccLayersInit()

    #初始化每个加速器上加速的层，都为0
    def getAccLayersInit(self):
        for acc in self.AccList:
            accName = acc.getAccName()
            self.AccLayers[accName]=[]
            self.translataftermapping[accName] = 0
            self.cost[accName] = 0
            self.acclastlayer[accName] = []

    # get the modalities and create two graph, Gmap to analyze the dependency, Gtime to keep the timeline in analyzing
    def getModalityLayers(self, LayerList):
        #将modalities赋值为layerlist
        self.modalities=LayerList
        #将modalities里所有的modality变成一个权重为data数量的权重图Gmap
        self.getLayers2Graph()
        #将上述Gmap复制一个作为Gtime
        self.Gtime=copy.deepcopy(self.Gmap)

    def getLayers2Graph(self):
        self.Gmap = nx.DiGraph()
        for modality in self.modalities:
            layers = list(modality.keys())
            # print(layers)
            layerIndex = 0
            lastNode = ''
            for layer in layers:
                LayerObj = modality[layer]
                # print(LayerObj)
                nodeName = LayerObj['LayerName']
                LayerWeight = self.getLayerWeight(LayerObj)
                self.Gmap.add_node(nodeName, attri=LayerObj, weight = LayerWeight, isLayer=True, isAssigned = False, acc = None, bind = False, start=0, end=0)
                if layerIndex == 0:
                    layerIndex = layerIndex + 1
                    lastNode = nodeName
                else:
                    self.Gmap.add_edge(lastNode, nodeName)
                    layerIndex = layerIndex + 1
                    lastNode = nodeName

    def getGraphVisual(self, Gname):
        if Gname == 'Gmap':
            nx.draw(self.Gmap, with_labels=True)
            plt.show()
        elif Gname == 'Gtime':
            nx.draw(self.Gtime, with_labels=True)
            plt.show()
        elif Gname == 'Gacc':
            nx.draw(self.Gacc, with_labels=True)
            plt.show()

    # add the assigned layers to acc object
    def getAccLayersAssign(self):
        touchedAcc = list(self.AccLayers.keys())
        for accName in touchedAcc:
            for acc in self.AccList:
                if acc.accName == accName:
                    acc.assiLayers = self.AccLayers[accName]

    # add the binding layers to acc object (layers that can store weights on Acc)
    def getAccLayersBind(self):
        for acc in self.AccList:
            # if this acc is assigned any layers
            if acc.assiLayers:
                # we need to see is there any storage left for newly assigned layer
                # acc.bindLayer is now the binded layers from previous modality
                accOccupy = 0
                for layer in acc.bindLayer:
                    layerInfo = self.getLayerInModality(layer)
                    acc.getRunPara(layerInfo)
                    if acc.accType == 'CNN':
                        layerWeiElement = acc.K*acc.K*acc.N*acc.M
                    elif acc.accType == 'LSTM':
                        layerWeiElement = acc.Hidden*4*(acc.Hidden+acc.Dimen)
                    elif acc.accType == 'FC':
                        layerWeiElement = acc.M*acc.N
                    accOccupy = accOccupy + layerWeiElement

                layerLeft = copy.deepcopy(acc.assiLayers)
                for layer in acc.bindLayer:
                    layerLeft.remove(layer)
                # lets see if we can add more layers to bind layers
                for layer in layerLeft:
                    layerInfo = self.getLayerInModality(layer)
                    acc.getRunPara(layerInfo)
                    if acc.accType == 'CNN':
                        layerWeiElement = acc.K*acc.K*acc.N*acc.M
                    elif acc.accType == 'LSTM':
                        layerWeiElement = acc.Hidden*4*(acc.Hidden+acc.Dimen)
                    elif acc.accType == 'FC':
                        layerWeiElement = acc.M*acc.N
                    if accOccupy + layerWeiElement <= acc.extraWeightElement:
                        acc.bindLayer.append(layer)
                        accOccupy = accOccupy + layerWeiElement
                    elif accOccupy + layerWeiElement > acc.extraWeightElement:
                        break

    def getLayerInModality(self, layer):
        for modality in self.modalities:
            if layer in list(modality.keys()):
                layerInfo = modality[layer]
                #layerInfo e.g {'LayerType': 'CNN', 'LayerName': 'M1L1', 'M': 66,  'N': 24, 'R': 55, 'C': 55, 'K': 11, 'S': 1, 'P': 1}
                return layerInfo

    def getAccmap(self, Accs):
        self.Amap = nx.DiGraph()
        cpus = {'cpu1', 'cpu2'}
        p2p = {}
        for cpu in cpus:
            self.Amap.add_node(cpu)
            for cput in cpus:
                if cput != cpu:
                    self.Amap.add_weighted_edges_from([(cput, cpu, 10)])
        for Acc in Accs:
            nodename =Acc.getAccName()
            self.Amap.add_node(nodename)
        for Acc in Accs:
            nodename = Acc.getAccName()
            p2p = Acc.getp2p(AccsList)
            host = Acc.getHostName()
            self.Amap.add_weighted_edges_from([(nodename, host, Acc.NetBand)])
            # for acc in Accs:
            #     accname = acc.getAccName()
            #     if accname != nodename:
            #         self.Amap.add_weighted_edges_from([(nodename, accname, p2p[accname])])

                # else:
                #     self.Amap.add_weighted_edges_from([(nodename, accname, float(100))])


    def getMapping(self):
        self.Gacc.clear()
        self.AccList = self.AccListNoTouch
        self.getAccLayersInit()
        self.getLayers2Graph()
        self.Gtime = copy.deepcopy(self.Gmap)
        Gmaptmp = copy.deepcopy(self.Gmap)
        AccInitTotLatency = self.getAccInitTotLatency(self.AccList)
        while self.getGraphSize(self.Gmap) > 0:
            layers2Map = self.getNoPredesorNode(self.Gmap)
            accs2Map = self.getAccCategory(self.AccList)
            for type in self.MappingType:
                currentTypeLayers = layers2Map[type]
                if currentTypeLayers:
                    # 如果加速器数量大于等于层数,就一次计算都用上加速器
                    if len(accs2Map[type]) >= len(layers2Map[type]):
                        # Accs2Map is list in list, permutations
                        # 不同的层在加速器中布置的各种组合
                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(layers2Map[type]))
                        layers2MapTmp = layers2Map[type]
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS1(layers2MapTmp, Accs2MapTmp,
                                                                                          AccInitTotLatency,
                                                                                          self.Gmap, self.Gtime,
                                                                                          self.Amap, Gmaptmp)

                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)
                    # 如果加速器数量小于层数，则同时计算层的数量为加速器数量
                    elif len(accs2Map[type]) < len(layers2Map[type]):
                        LayersComb = self.getComb(layers2Map[type], len(accs2Map[type]))

                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(accs2Map[type]))
                        layers2MapTmp = LayersComb
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS2(layers2MapTmp, Accs2MapTmp,
                                                                                          AccInitTotLatency,
                                                                                          self.Gmap, self.Gtime,
                                                                                          self.Amap, Gmaptmp)

                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)

                        # update the graph and self.AccLayers

                else:
                    continue

        # check if Gmap is empty and if all nodes in Gtime is assigned an acc
        if not self.getGraphValidate(self.Gmap, self.Gtime):
            print('the graph Gmap is not cleaned or Gtime is not built correctly')
            exit()

        # save self.Gtime and self.AccLayers to self.GtimeLastTune and self.AccLayersLastTune
        # self.Gacc is not built yet
        self.GtimeReserve = copy.deepcopy(self.Gtime)
        self.AccLayersReserve = copy.deepcopy(self.AccLayers)
        # self.AccListReserve should always be clean, without any attri
        self.AccListReserve = copy.deepcopy(self.AccList)
        print(self.AccLayers)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        power = self.getSysPower(self.Gtime)

        # transLat = self.getJointDNNTrans(self.modalities, self.NetBand, self.Dtype)
        # print('after mapping, the system elapsed time is: ', self.PlatformElapsed)
        # print('after mapping, the system energy is: ', power)
        # print('after mapping, the system accumulated execution time is: ', accTimeSum)
        # print('after mapping, the system accumulated trans time is: ', transLat)

        return

        # return self.AccLayers, AccInitTotLatency

    def getMappingcom(self):
        # 有Gmap和Gtime
        # InitMapper.getModalityLayers(Modality1)
        AccInitTotLatency = self.getAccInitTotLatency(self.AccList)
        Gmaptmp = copy.deepcopy(self.Gmap)
        # Gmapstart = copy.deepcopy(self.Gmap)
        self.Gtime  = copy.deepcopy(self.Gmap)

        accs2Map = self.getAccCategory(self.AccList)

        self.Gtime = copy.deepcopy(self.Gmap)
        # Gmapstarttmp = copy.deepcopy(Gmapstart)

        for acc in self.AccList:
            accName = acc.getAccName()
            self.AccLatencymap[accName] = []
            self.acclastlayer[accName] = []

        iteration = 0
        i = 0
        layers2Map = {}
        while self.getGraphSize(Gmaptmp) > 0:
            layers2Map[i] = []
             # 下面是对一群初始节点的计算
            for type in self.MappingType:
                starters = self.getNoPredesorNode(Gmaptmp)
                currentTypeLayers = starters[type]
                if currentTypeLayers:
                    distance = float('inf')
# **********************************************************************************************************************
#                     edition2
                    layer2acc = self.getPerRep(accs2Map[type], min(len(accs2Map[type]), len(starters[type])))
                    startersTmp = self.getComb(starters[type], min(len(accs2Map[type]), len(starters[type])))

                    layers2Maptmp = []
# **********************************************************************************************************************
#                     edition2
#                     layerTimeMapped, AccInitTotLatency, layers2Maptmp, iteration, self.acclastlayer = self.getMinLatency(startersTmp, layer2acc,
#                                                                                            self.Amap, AccInitTotLatency,
#                                                                                            Gmaptmp, self.Gtime, iteration, self.acclastlayer)
                    layerTimeMapped, AccInitTotLatency, layers2Maptmp, iteration, self.cost, self.acclastlayer = self.getMinLatency1(startersTmp, layer2acc,
                                                                                           self.Amap, AccInitTotLatency,
                                                                                           Gmaptmp, self.Gtime, iteration, self.cost, self.acclastlayer)
                    layers2Map[i] = layers2Maptmp
                    # print(layerTimeMapped)
                    # for k, v in layerTimeMapped.items():
                    #     # predecessor = self.AccLatencymap.predecessors(k)
                    #     self.AccLatencymap[v[0]].append(k)
                    # print(self.AccLatencymap)
                    self.AccLayers, Gmaptmp, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                             Gmaptmp, self.Gtime)

                else:
                    continue

                i += 1

        self.Gtimeaftermapping = copy.deepcopy(self.Gtime)
        self.Acctimeaftermapping = copy.deepcopy(AccInitTotLatency)

        # 此处是获得通讯时间
        nodes = list(self.Gtime)
        complat = {}
        translat = {}
        for acc in self.AccList:
            acc = acc.getAccName()
            translat[acc] = 0
            complat[acc] = 0
        for node in nodes:
            layerInfo = self.Gtime.nodes[node]['attri']
            accel = self.Gtime.nodes[node]['acc']
            latency = self.Gtimeaftermapping.nodes[node]['end'] - self.Gtimeaftermapping.nodes[node]['start']
            for acc in self.AccList:
                accname = acc.getAccName()
                if accel == accname:
                    acc.getRunPara(layerInfo)
                    Lat, Th = acc.getLayerPerf1(WEItrans=False, IFMtrans=False, OFMtrans=False)
                    complat[accel] += (Lat)
                    translat[accel] += latency - (Lat)
                else:
                    complat[accel] = complat[accel]
                    translat[accel] = translat[accel]

        self.cost = {}
        self.acclastlayer = {}
        self.translataftermapping = copy.deepcopy(translat)
        self.complataftermapping = copy.deepcopy(complat)
        self.layers2Map = copy.deepcopy(layers2Map)
        if not self.getGraphValidate(Gmaptmp, self.Gtime):
            print('the graph Gmap is not cleaned or Gtime is not built correctly')
            exit()

        self.GtimeReserve = copy.deepcopy(self.Gtime)
        self.AccLayersReserve = copy.deepcopy(self.AccLayers)
        self.AccListReserve = copy.deepcopy(self.AccList)
        print('accelerator allocation:')
        print(self.AccLayers)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        self.timeaftermapping = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        power = self.getSysPower(self.Gtime)
        self.energyaftermapping = self.getSysPower(self.Gtime)

        # print('after mapping, the system elapsed time is: ', self.PlatformElapsed)
        # print('after mapping, the system energy is: ', power)
        return

    def getLayerOrder(self):
        #得到交换Gmap
        self.GmapPermu = copy.deepcopy(self.Gmap)
        #如果图中的nodes数量大于零
        while self.getGraphSize(self.GmapPermu) > 0:
            #得到GmapPermu中所有无前位节点的node
            layers2Map = self.getNoPredesorNode(self.GmapPermu)
            # mapping contains the types of layer to be mapped: CNN/LSTM, or even more
            avaLayers = []
            #mappingtype包括CNN，LSTM，FCN
            for type in self.MappingType:
                # 所有无前位节点的不同mappingtpyelist
                currentTypeLayers = layers2Map[type]
                avaLayers = avaLayers + currentTypeLayers
            self.LayerOrder.append(avaLayers[0])
            #将此modality中的无前位节点按CNN LSTM FCN的顺序拍起来
            self.GmapPermu.remove_node(avaLayers[0])


    def getNodeShortenBinding(self, node):
        # 得到当前节点的所在加速器的前一层和所在神经网络的前一层
        HWpreds = list(self.Gacc.predecessors(node))
        SWpreds = list(self.Gtime.predecessors(node))

        preds = HWpreds + SWpreds
        # print(1)
        # print(node)
        # print(preds)
        # 得到加速器对前一层计算时间完成的时间，神经网络的前一层计算完成的时间
        # 该层的开始时间应该是这两者中较大的一个
        maxend = 0
        for pred in preds:
            if self.Gacc.nodes[pred]['end'] > maxend:
                maxend = self.Gacc.nodes[pred]['end']

        # 如果该层应该的开始时间小于其初次map的开始时间
        if maxend < self.Gacc.nodes[node]['start']:
            # 获得该层的计算时间
            latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
            # 重置该层的开始时间为(加速器或者神经网络)前一层的终止时间
            self.Gacc.nodes[node]['start'] = maxend
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
            # 将时间图的开始时间和结束时间都重置为经过修正的时间
            self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
            self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
            # 继续得到当前层的加速器和神经网络的后续层
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            # will the recursion be ended by the graph it self? as the last node will not have successor
            # 直到当前计算的层没有后续层，终止迭代
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)
        else:
            return

    def getNodeShortenBindingLastTune(self, node):
        HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
        SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))

        preds = HWpreds + SWpreds
        maxend = 0
        for pred in preds:
            if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                maxend = self.GaccLastTuneTmp.nodes[pred]['end']

        if maxend != self.GaccLastTuneTmp.nodes[node]['start']:
            latency = self.GaccLastTuneTmp.nodes[node]['end'] - self.GaccLastTuneTmp.nodes[node]['start']
            self.GaccLastTuneTmp.nodes[node]['start'] = maxend
            self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + latency
            self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
            self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
            HWsuccessor = self.GaccLastTuneTmp.successors(node)
            SWsuccessor = self.GtimeLastTuneTmp.successors(node)
            # will the recursion be ended by the graph it self? as the last node will not have successor
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBindingLastTune(HWsu)
            # for SWsu in list(SWsuccessor):
            #     self.getNodeShortenBindingLastTune(SWsu)
        else:
            return

    def getHelperGaccTime(self):
        for acc in self.AccList:
            layers = list(acc.assiLayers)
            print('timeline of acc:', acc.accName)
            self.getGraphTimePrint(self.Gacc, layers)
        print('end Gacc graph time show************************************************************')

    def getKnapsack(self, BindLayers= None):
        # we will solve a Knapsack problem problem, select layers in to weight storage
        # we dont change any attri in self.Gtime in self.getMap2AccObjandGtime
        # we only change self.AccList to add assign and bind layer to a accObj
        self.AccList, self.Gtime = self.getMap2AccObjandGtime(self.AccLayers, self.AccList, self.Gtime, BindLayers)

        # just build Gacc graph, does not do any time analysis
        self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        # print(self.Gacc.nodes)
        self.GaccReserve= copy.deepcopy(self.Gacc)

    def getBindedTimeH2H(self):
        self.getLayers2Graph()
        # lets do a  check to see timegraph and acc graph attribute
        # print(self.Gtime, self.Gacc)
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        # self.getHelperGaccTime()
        GaccTmp=copy.deepcopy(self.Gacc)
        # break condition, the GaccTmp does not contain any Binded layer
        # at the end of each iteration, remove the selected bind layer and its successor in GaccTmp, update the self.Gacc
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallBindLayerinGraph(GaccTmp)
            # now update the Gacc graph time, also the Gtime graph time
            layerInfo = self.Gacc.nodes[node]['attri']
            # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
            # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
            acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            nodeaccname = acc.getAccName()
            acc.getRunPara(layerInfo)
            # LatSec, Th = acc.getLayerPerf(WEItrans=False)

            predecessors = list(self.Gtime.predecessors(node))
            tDataInfromMem = 0
            tDataOutfromMem = 0
            if predecessors:
                for predecessor in predecessors:
                    predaccname = self.Gtime.nodes[predecessor]['acc']
                    if nodeaccname == predaccname:
                        bandwidth = float('inf')
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=False,
                                                                OFMtrans=False)
                    else:
                        bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=True, OFMtrans=False)
                    tDataInfromMem += tDataIn
                    tDataOutfromMem += tDataOut
                LatSec, Th = acc.getLayerPerf1(WEItrans=False, IFMtrans=False, OFMtrans=False)
                LatSec += (tDataInfromMem + tDataOutfromMem)
            else:
                LatSec, Th = acc.getLayerPerf(WEItrans=False, IFMtrans=True, OFMtrans=False)

            # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start']+ LatSec
            self.Gtime.nodes[node]['end'] = self.Gtime.nodes[node]['start'] + LatSec
            # print('this layer is shortened to: ', self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start'])
            # then we should start the recursion process, consider the rest node
            # 此处在将某一节点的权重bind后，更新相邻节点的时间
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)
            # self.getHelperGaccTime()
        # print('This is time after getting weight binding!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()
        # **************************************************************************************************************
        # 此处是获得bind后的每个加速器的计算终止时间
        nodes = list(self.Gacc)
        for acc in self.AccList:
            accName = acc.getAccName()
            time = 0
            self.Acctimeafterbinding[accName] = 0
            for node in nodes:
                if self.Gacc.nodes[node]['acc'] == accName:
                    if self.Gacc.nodes[node]['end'] >= time:
                        self.Acctimeafterbinding[accName] = self.Gacc.nodes[node]['end']
                        time = self.Gacc.nodes[node]['end']
        # print(self.Acctimeafterbinding)

        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        # print('after binding, the system elapsed time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.Gacc)
        self.energyafterbinding = power
        # print('after binding, the system energy is: ', power)
        #
        # accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        # sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print('after binding, the system CTC time is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        return

    def getBindedTime(self):
        self.getLayers2Graph()
        # lets do a  check to see timegraph and acc graph attribute
        # print(self.Gtime, self.Gacc)
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        # self.getHelperGaccTime()
        GaccTmp=copy.deepcopy(self.Gacc)
        # break condition, the GaccTmp does not contain any Binded layer
        # at the end of each iteration, remove the selected bind layer and its successor in GaccTmp, update the self.Gacc
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallBindLayerinGraph(GaccTmp)
            # now update the Gacc graph time, also the Gtime graph time
            layerInfo = self.Gacc.nodes[node]['attri']
            # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
            # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
            acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            nodeaccname = acc.getAccName()
            acc.getRunPara(layerInfo)
            # LatSec, Th = acc.getLayerPerf(WEItrans=False)

            predecessors = list(self.Gtime.predecessors(node))
            tDataInfromMem = 0
            tDataOutfromMem = 0
            if predecessors:
                for predecessor in predecessors:
                    predaccname = self.Gtime.nodes[predecessor]['acc']
                    if nodeaccname == predaccname:
                        bandwidth = float('inf')
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=False,
                                                                OFMtrans=False)
                    else:
                        bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=True, OFMtrans=False)
                    tDataInfromMem += tDataIn
                    tDataOutfromMem += tDataOut
                LatSec, Th = acc.getLayerPerf1(WEItrans=False, IFMtrans=False, OFMtrans=False)
                LatSec += (tDataInfromMem + tDataOutfromMem)
            else:
                LatSec, Th = acc.getLayerPerf(WEItrans=False, IFMtrans=True, OFMtrans=False)

            # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start']+ LatSec
            self.Gtime.nodes[node]['end'] = self.Gtime.nodes[node]['start'] + LatSec
            # print('this layer is shortened to: ', self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start'])
            # then we should start the recursion process, consider the rest node
            # 此处在将某一节点的权重bind后，更新相邻节点的时间
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)
            # self.getHelperGaccTime()
        # print('This is time after getting weight binding!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()
        # **************************************************************************************************************
        # 此处是获得bind后的每个加速器的计算终止时间
        nodes = list(self.Gacc)
        for acc in self.AccList:
            accName = acc.getAccName()
            time = 0
            self.Acctimeafterbinding[accName] = 0
            for node in nodes:
                if self.Gacc.nodes[node]['acc'] == accName:
                    if self.Gacc.nodes[node]['end'] >= time:
                        self.Acctimeafterbinding[accName] = self.Gacc.nodes[node]['end']
                        time = self.Gacc.nodes[node]['end']
        # print(self.Acctimeafterbinding)

        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        print('after binding, the system elapsed time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.Gacc)
        self.energyafterbinding = power
        print('after binding, the system energy is: ', power)
        #
        # accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        # sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print('after binding, the system CTC time is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        return

    def getNodeExtend(self, node):
        HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
        SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))
        earlyStart = 0
        for HWpred in HWpreds:
            if self.GaccLastTuneTmp.nodes[HWpred]['end'] > earlyStart:
                earlyStart = self.GaccLastTuneTmp.nodes[HWpred]['end']
        for SWpred in SWpreds:
            if self.GtimeLastTuneTmp.nodes[SWpred]['end'] > earlyStart:
                earlyStart = self.GtimeLastTuneTmp.nodes[SWpred]['end']


        length = self.GaccLastTuneTmp.nodes[node]['end']-self.GaccLastTuneTmp.nodes[node]['start']
        self.GaccLastTuneTmp.nodes[node]['start'] = earlyStart
        self.GaccLastTuneTmp.nodes[node]['end'] = earlyStart + length

        self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

        HWsuccessor = self.GaccLastTuneTmp.successors(node)
        SWsuccessor = self.GtimeLastTuneTmp.successors(node)
        # will the recursion be ended by the graph it self? as the last node will not have successor
        for HWsu in list(HWsuccessor):
            self.getNodeExtend(HWsu)
        # for SWsu in list(SWsuccessor):
        #     self.getNodeExtend(SWsu)
        if not list(HWsuccessor) :
            return
        # if not list(SWsuccessor):
        #     return

    def getGraphUpdatViaModalLastTune(self, node):
        # 在一个acc上 相同modality layer且连续 那么这团layer时间整体缩短
        # group的第一个acc 起始时间暂时不改变 由另一个函数去做全局修改
        # 用递归的形式遍历每一个点， 分情况讨论，修改这个点的 start end time
        # warning, 就算每个layer跟前面是相同modality,也要看是不是sw connected的！ 避免出错！
        # Gacc 一定最多有且只有一个pred或者success
        preds = list(self.GaccLastTuneTmp.predecessors(node))
        successs = list(self.GaccLastTuneTmp.successors(node))

        #special case init node and last node, or the node is init and last node
        #the node is init and last node, return, no change should be done
        if not preds and not successs:
            return
        #the node is init, it can has the same modal with its successor or not
        if not preds and successs:
            success = successs[0]
            # the node is init, it can has the same modal with its successor
            # X (red) red
            if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # X (red) red, the two red not continuous
            elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)
            # the node is init, it does not has the same modal with its successor
            # X (red) green
            elif not self.getModalCheck(node, success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

        #the node is last node
        if preds and not successs:
            pred=preds[0]
            # the node is last, it can has the same modal with its pred
            # red (red) X
            if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # end recursion
                return
            # red (red) X, two red not continuous
            if self.getModalCheck(pred, node) and not self.getDependCheck(pred, node):
                # end recursion
                return

            # the node is last, it can has different modal with its pred
            # blue (red) X
            elif not self.getModalCheck(pred, node):
                # do nothing, end recursion
                return

        if preds and successs:
            pred=preds[0]
            success = successs[0]
            # (layer) is our target
            # case 1-0: red (red) red
            if self.getModalCheck(node,success) and self.getModalCheck(pred,node) and \
                    self.getDependCheck(node,success) and self.getDependCheck(pred,node):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case 1-1: red (red) red, first two are continuous, last two not
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success) and self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case 1-2: red (red) red, first two are not continuous, last two are
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case 1-3: red (red) red, they are not continuous
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)


            # case: green (red) red
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    self.getDependCheck(node, success):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case: green (red) red, but two red are not continuous
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case: red (red) green
            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

                # case: red (red) green, the two red are not continuous
            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(pred, node):
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

            # case: blue (red) green
            elif not self.getModalCheck(node, success) and not self.getModalCheck(pred, node):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModalLastTune(success)

    def getTunePerf(self,node,neighbor, relation):
        # 此处得到神经网络相邻层之间如果加速器相同，
        # we need to update both self.GtimeLastTune, self.GaccLastTune, self.AccLayersLastTune
        # a node change acc assignment to its pred
        self.GtimeLastTuneTmp = copy.deepcopy(self.GtimeReserve)
        self.GaccLastTuneTmp = copy.deepcopy(self.GaccReserve)
        self.AccLayersLastTuneTmp = copy.deepcopy(self.AccLayersReserve)
        # self.GtimeLastTuneTmp是神经网络的图
        # DiGraph with 10 nodes and 7 edges
        # self.GaccLastTuneTmp是加速器的map图
        # DiGraph with 10 nodes and 8 edges
        # self.AccLayersLastTuneTmp是加速器上面map的图的配置
        # {'cnn_acc_1': ['M1L1', 'M2L1', 'M2L2', 'M1L3', 'M2L4'], 'cnn_acc_2': ['M1L2', 'M2L3', 'M3L1', 'M3L2', 'M3L3']}
        # 如果neighbor是node的前位节点
        if relation == 'pred':
            # 记录当前节点和邻居节点的加速器序号
            nodeAcc = self.GtimeLastTuneTmp.nodes[node]['acc']
            predAcc = self.GtimeLastTuneTmp.nodes[neighbor]['acc']
            # change the assignment attribute
            # 将当前层的加速器类型改成前位层的加速器类型
            self.GtimeLastTuneTmp.nodes[node]['acc'] = predAcc
            self.GaccLastTuneTmp.nodes[node]['acc'] = predAcc
            self.AccLayersLastTuneTmp[nodeAcc].remove(node)
            predAccPos =  self.AccLayersLastTuneTmp[predAcc].index(neighbor)+1
            self.AccLayersLastTuneTmp[predAcc].insert(predAccPos, node)
            # change the connections in self.GtimeLastTune, self.GaccLastTune
            # now we need to update self.GtimeLastTune, self.GaccLastTune since we insert and remove a layer in two acc respectively
            # we dont need to change self.GtimeLastTune's edge
            # find pred's successor in self.GaccLastTune, unlink the pred and successor
            # 因为改变了神经网络里连续的层的映射，所以需要更新GaccLastTune，将改变的层的在加速器上的前后层断开
            # 然后将那两层连上
            predSuccOnAcc = list(self.GaccLastTuneTmp.successors(neighbor))
            nodesuccOnAcc = list(self.GaccLastTuneTmp.successors(node))
            nodepredOnAcc = list(self.GaccLastTuneTmp.predecessors(node))
            # 将node添加到pred所在的acc上
            # 需要判断predSuccOnAcc存不存在
            # 存在
            if predSuccOnAcc:
                predSuccOnAcc= predSuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(neighbor,predSuccOnAcc)
                self.GaccLastTuneTmp.add_edge(neighbor, node)
                self.GaccLastTuneTmp.add_edge(node, predSuccOnAcc)
            elif not predSuccOnAcc:
                self.GaccLastTuneTmp.add_edge(neighbor, node)
                # find node's successor and predecessor in self.GaccLastTune, link the pred and successor
                # 需要判断nodesuccOnAcc， nodepredOnAcc存不存在
            #将node从之前所在的acc上移除
            # 讨论node on acc的邻居是否存在
            if nodesuccOnAcc and nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
                self.GaccLastTuneTmp.add_edge(nodepredOnAcc, nodesuccOnAcc)
            elif not nodesuccOnAcc and nodepredOnAcc:
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
            elif nodesuccOnAcc and not nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
            elif not nodesuccOnAcc and not nodepredOnAcc:
                pass

            self.getNodeExtend(node)
            if nodesuccOnAcc:
                self.getNodeShortenBindingLastTune(nodesuccOnAcc)

        elif relation == 'success':
            nodeAcc = self.GtimeLastTuneTmp.nodes[node]['acc']
            successAcc = self.GtimeLastTuneTmp.nodes[neighbor]['acc']
            # change the assignment attribute
            self.GtimeLastTuneTmp.nodes[node]['acc'] = successAcc
            self.GaccLastTuneTmp.nodes[node]['acc'] = successAcc
            self.AccLayersLastTuneTmp[nodeAcc].remove(node)
            successAccPos = self.AccLayersLastTuneTmp[successAcc].index(neighbor)
            self.AccLayersLastTuneTmp[successAcc].insert(successAccPos, node)
            # change the connections in self.GtimeLastTune, self.GaccLastTune
            # now we need to update self.GtimeLastTune, self.GaccLastTune since we insert and remove a layer in two acc respectively
            # we dont need to change self.GtimeLastTune's edge
            # find successor's pred in self.GaccLastTune, unlink the pred and successor
            succPredOnAcc = list(self.GaccLastTuneTmp.predecessors(neighbor))
            # find node's successor and predecessor in self.GaccLastTune, link the pred and successor later
            nodesuccOnAcc = list(self.GaccLastTuneTmp.successors(node))
            nodepredOnAcc = list(self.GaccLastTuneTmp.predecessors(node))
            if succPredOnAcc:
                succPredOnAcc = succPredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(succPredOnAcc, neighbor)
                self.GaccLastTuneTmp.add_edge(succPredOnAcc, node)
                self.GaccLastTuneTmp.add_edge(node, neighbor)
            elif not succPredOnAcc:
                self.GaccLastTuneTmp.add_edge(node, neighbor)

            # 讨论node on acc的邻居是否存在
            if nodesuccOnAcc and nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
                self.GaccLastTuneTmp.add_edge(nodepredOnAcc, nodesuccOnAcc)
            elif not nodesuccOnAcc and nodepredOnAcc:
                nodepredOnAcc = nodepredOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(nodepredOnAcc, node)
            elif nodesuccOnAcc and not nodepredOnAcc:
                nodesuccOnAcc = nodesuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(node, nodesuccOnAcc)
            elif not nodesuccOnAcc and not nodepredOnAcc:
                pass

            # 完成后更新时间
            self.getNodeExtend(node)
            if nodesuccOnAcc:
                self.getNodeShortenBindingLastTune(nodesuccOnAcc)

        # dont do anything to self.AccList. every time we use a clean one. after a iteration, we
        # get a self.Acclist with arribute defined
        self.GaccReserveTmp = self.GaccLastTuneTmp
        self.GtimeReserveTmp = self.GtimeLastTuneTmp
        self.AccLayersReserveTmp = self.AccLayersLastTuneTmp


        # self.AccListLastTuneTmp, self.GtimeLastTuneTmp = self.getMap2AccObjandGtime(self.AccLayersLastTuneTmp,self.AccListLastTuneTmp,self.GtimeLastTuneTmp, None)
        self.getBindedTimeLastTune()
        self.getIfmOfmTransLastTune()

    def getHomoNeighbor(self):
        # self.GtimeLastTune, self.AccLayersLastTune from getMapping
        # self.GaccLastTune from getBindedTime
        # they are result for self.getMapping
        # 得到神经网络图中的所有的最后的层
        lastNode = []
        for node in list(self.Gtime.nodes):
            if len(list(self.Gtime.successors(node))) == 0:
                lastNode.append(node)
        # 指定按字典里的value长度排序，同时进行翻转，即按value的长度从大到小排列
        # 此处是按照终止节点的计算终止时间从大到小排序（reverse=True）
        lastNode = sorted(lastNode, key=lambda x: self.Gtime.nodes[x]['end'], reverse=True)
        # print('******************************************************************************************')
        # print(lastNode)
        initNode = copy.deepcopy(lastNode)
        # 按计算完成时间从大到小得到神经网络的初始层initNode
        for i in range(len(initNode)):
            element = initNode[i]
            element = element.split("L")
            element = element[0] + 'L1'
            initNode[i] = element
        # print(initNode)
        # print('******************************************************************************************')
        # print(self.AccLayersReserve)

        # self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        # print('before color flip, the system elapsed time is: ', self.PlatformElapsed)

        self.GaccLastTune = copy.deepcopy(self.Gacc)
        self.GtimeLastTune = copy.deepcopy(self.Gtime)
        self.AccLayersLastTune = copy.deepcopy(self.AccLayers)
        self.AccListLastTune = copy.deepcopy(self.AccList)

        # we want to flip the node color in the graph multiple times
        initNodeRepeat = []
        for i in range(2):
            initNodeRepeat = initNodeRepeat + initNode
        # initNodeRepeat是初始节点的double
        # ['M3L1', 'M2L1', 'M1L1', 'M3L1', 'M2L1', 'M1L1']

        # 此处是神经网络相邻层的加速器类型转化
        start = time.time()
        for node in initNodeRepeat:
            # 此处是将神经网络的相邻层的加速器的种类设置成一样然后进行计算
            self.getModalTune(node)
        end = time.time()
        timeElapse = end - start

        # 此处是所有层计算完成的时间,不算并行计算，是每个神经网络的每个层的计算时间的和
        accTimeSum = self.getAccsAccuTimeSum(self.GaccLastTune)
        # 此处是某一个神经网络的计算完成时间
        sysCompLat = self.getJointDNNCompLat(self.AccList, self.GaccLastTune)
        # 此处是加速器完成所有计算的时间
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.GaccLastTune))
        # print(self.GaccLastTune)
        print(self.AccLayersReserve)
        print('after color flip, the system elapsed time is: ', self.PlatformElapsed)
        # print('after color flip, the system CTC is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        power = self.getSysPower(self.GaccLastTune)
        print('after color flip, the system energy is: ', power)
        # print('the color flip search time is : ', timeElapse)

        # print('time check')
        # nodes = list(self.GtimeLastTune.nodes())
        # for node in nodes:
        #     print(self.GtimeLastTune.nodes[node]['bind'])

        print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def getBindedTimeLastTune(self):
        # we will solve a Knapsack problem problem, select layers in to weight storage
        self.AccListLastTuneTmp, self.GtimeLastTuneTmp = self.getMap2AccObjandGtime(self.AccLayersLastTuneTmp, self.AccListReserve, self.GtimeLastTuneTmp, None)
        self.getLayerTimeVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        # save self.Acclist to self.AccListLastTune, for last tune
        self.GaccLastTuneTmp = self.getAccTimeGraph(self.AccListLastTuneTmp, self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        # lets do a grap check to see timegraph and acc graph attribute
        self.getLayerBindVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        self.getLayerTimeVarify(self.GtimeLastTuneTmp, self.GaccLastTuneTmp)
        # self.getHelperGaccTime()
        GaccTmp=copy.deepcopy(self.GaccLastTuneTmp)
        # break condition, the GaccTmp does not contain any Binded layer
        # at the end of each iteration, remove the selected bind layer and its successor in GaccTmp, update the self.Gacc
        # 更新时间
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallBindLayerinGraph(GaccTmp)
            acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
            layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
            acc.getRunPara(layerInfo)
            nodeaccname = acc.getAccName()
            # now update the Gacc graph time, also the Gtime graph time
            # ******************************************************************
            predecessors = list(self.GtimeLastTuneTmp.predecessors(node))
            # successors = list(self.Gtime.successors(layerName))
            if predecessors:
                for predecessor in predecessors:
                    predaccname = self.GtimeLastTuneTmp.nodes[predecessor]['acc']
                    if predaccname == nodeaccname:
                        bandwidth = float('inf')
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=True, IFMtrans=False,
                                                                OFMtrans=False)
                    else:
                        bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=True, IFMtrans=False,
                                                                OFMtrans=False)
            else:
                tDataIn, tDataOut = acc.getDataInPerf(WEItrans=True, IFMtrans=False,
                                                      OFMtrans=False)
            LatSec, Th = acc.getLayerPerf1()
            LatSec += (tDataIn + tDataOut)
            # ******************************************************************
            # layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
            # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
            # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])

            # LatSec, Th = acc.getLayerPerf(WEItrans=False)
            # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
            self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start']+ LatSec
            self.GtimeLastTuneTmp.nodes[node]['end'] = self.GtimeLastTuneTmp.nodes[node]['start'] + LatSec
            # print('this layer is shortened to: ', self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start'])
            # then we should start the recursion process, consider the rest node
            HWsuccessor = self.GaccLastTuneTmp.successors(node)
            SWsuccessor = self.GtimeLastTuneTmp.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBindingLastTune(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBindingLastTune(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)

        return

    def getGraphUpdatViaModal(self, node):
        # 在一个acc上 相同modality layer且连续 那么这团layer时间整体缩短
        # group的第一个acc 起始时间暂时不改变 由另一个函数去做全局修改
        # 用递归的形式遍历每一个点， 分情况讨论，修改这个点的 start end time
        # warning, 就算每个layer跟前面是相同modality,也要看是不是sw connected的！ 避免出错！
        # Gacc 一定最多有且只有一个pred或者success
        # Gacc是加速器的前后
        preds = list(self.Gacc.predecessors(node))
        successs = list(self.Gacc.successors(node))

        #special case init node and last node, or the node is init and last node
        #the node is init and last node, return, no change should be done
        # 如果没有前位节点后后续节点
        if not preds and not successs:
            return
        #the node is init, it can has the same modal with its successor or not
        # 无前位节点但是有后续节点
        if not preds and successs:
            success = successs[0]
            # the node is init, it can has the same modal with its successor
            # X (red) red
            # 检查此加速器上的该层及其后续层是否是同一个modality的连续的两层
            if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                # AccList是[<cnn_acc_1.cnn_acc_1 object at 0x01B42030>, <cnn_acc_2.cnn_acc_2 object at 0x01B422F0>]
                # self.Gacc.nodes[node]['acc']是cnn_acc_2
                # accObj是cnn_acc_2
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo
                # self.Gacc 是DiGraph with 10 nodes and 8 edges
                # print(self.Gacc.edge)
                # nx.draw(self.Gacc, node_color='y', with_labels=True, node_size=500)
                # self.Gacc.nodes[node]是一个层的所有信息，
                # 包括attri(计算信息)，weight，isLayer，isAssigned，所在acc，权重是否bind，开始和终止时间
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                # 如果bind，就不用传输权重
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # 获取该层的计算信息
                # accObj.getRunPara(layerInfo)
                # 计算该层的计算延迟的throughput
                # 因为是开始的节点，所以有IFMtrans
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                # 更新时间
                # 只要更新结束时间，开始时间都是不受影响，因为每次更新的节点是无前位节点
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # print(3)
                # print(node)
                # recursion, the next node
                # 继续更新该层的后一层
                self.getGraphUpdatViaModal(success)

            # X (red) red, the two red not continuous
            # 如果加速器上的当前层和后一层属于同一个神经网络，但是不是连续的
            elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModal(success)
            # the node is init, it does not has the same modal with its successor
            # X (red) green
            elif not self.getModalCheck(node, success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModal(success)

        #the node is last node
        # 无后续节点，只有前位节点的最后一层
        if preds and not successs:
            pred=preds[0]
            # the node is last, it can has the same modal with its pred
            # red (red) X
            # 如果最后一层和其所在加速器的前一层为同一个神经网络的连续层，则往下更新
            if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # # 因为是最后的节点，所以有OFMtrans
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                # 本层的开始时间为前位节点的结束时间
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # end recursion
                return
            # red (red) X, two red not continuous
            # 如果是同一个神经网络但是不连续，则不动。
            if self.getModalCheck(pred, node) and not self.getDependCheck(pred, node):
                # end recursion
                return

            # the node is last, it can has different modal with its pred
            # blue (red) X
            # 如果不是同一个神经网络，则不动
            elif not self.getModalCheck(pred, node):
                # do nothing, end recursion
                return

        # 如果是当前层既有前位层也有后续层
        if preds and successs:
            pred=preds[0]
            success = successs[0]
            # (layer) is our target
            # case 1-0: red (red) red
            # 如果和前位和后续都是同一个神经网络的连续层
            if self.getModalCheck(node,success) and self.getModalCheck(pred,node) and \
                    self.getDependCheck(node,success) and self.getDependCheck(pred,node):
                # 得到此层加速器的名字
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModal(success)
            # 如果该层和前后层是同一个神经网络，但只和前层连续
            # case 1-1: red (red) red, first two are continuous, last two not
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success) and self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # 如果该层和前后层是同一个神经网络，但只和后层连续
            # case 1-2: red (red) red, first two are not continuous, last two are
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case 1-3: red (red) red, they are not continuous
            # 如果该层和前后层是同一个神经网络，但都不连续
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # 该层和后一层是同一个神经网络且连续，与前位层不属于一个神经网络
            # case: green (red) red
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    self.getDependCheck(node, success):
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # 该层和后一层是同一个神经网络但不连续，与前位层不属于一个神经网络
            # case: green (red) red, but two red are not continuous
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # 该层和前一层是同一个神经网络且连续，与后层不属于一个神经网络
            # case: red (red) green
            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(pred, node):
                # accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                # layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                # accObj.getRunPara(layerInfo)
                # LatSec, Th = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                acc.getRunPara(layerInfo)
                nodeaccname = acc.getAccName()
                # now update the Gacc graph time, also the Gtime graph time
                # ******************************************************************
                predecessors = list(self.Gtime.predecessors(node))
                # successors = list(self.Gtime.successors(layerName))
                if predecessors:
                    for predecessor in predecessors:
                        predaccname = self.Gtime.nodes[predecessor]['acc']
                        if predaccname == nodeaccname:
                            bandwidth = float('inf')
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                        else:
                            bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                            tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=WEItrans, IFMtrans=False,
                                                                    OFMtrans=False)
                else:
                    tDataIn, tDataOut = acc.getDataInPerf(WEItrans=WEItrans, IFMtrans=False,
                                                          OFMtrans=False)
                LatSec, Th = acc.getLayerPerf1()
                LatSec += (tDataIn + tDataOut)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # 该层和前一层是同一个神经网络但不连续，与后层不属于一个神经网络
            # case: red (red) green, the two red are not continuous
            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(pred, node):
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case: blue (red) green
            elif not self.getModalCheck(node, success) and not self.getModalCheck(pred, node):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModal(success)


    def getIfmOfmTrans(self):
        # initnode is the starting layer of a acc
        # initnode是某个加速器的初始层
        # 如果该节点的predecessor为0，则说明是初始节点
        # print(self.Gacc)
        # self.Gacc.nodes是按加速器顺序排的层
        initNode = []
        for node in list(self.Gacc.nodes):
            if len(list(self.Gacc.predecessors(node)))==0:
                initNode.append(node)
        # tune each acc's layer time according to modality similarity
        # 如果一个加速器上的相邻层是同一个神经网络的连续层，那么缩短时间
        # 修改Gacc图的起始时间和终止时间
        for node in initNode:
            self.getGraphUpdatViaModal(node)
        # 此时缩短了一坨坨相同modal的layer的总体时间 但是graph整体时间还没有去缩短
        # print('This is time after modality similarity!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()

        # check the first node of Gacc, if its the first node in SW, removed, it can not be shortened
        GaccTmp = copy.deepcopy(self.Gacc)
        for node in initNode:
            # layerOrder现在是每一个加速器上当前计算的层
            # ['M1', '1']
            layerOrder = node.split("L")
            # 如果加速器当前计算的层是某一个神经网络的第一层，则无法加速，从GaccTmp中移除该层
            if layerOrder[1] == '1':
                GaccTmp.remove_node(node)
            else:
                continue
            # 最后GaccTmp是加速器的当前计算的层中所有可以加速的层
        # break condition, the GaccTmp does not contain any layer
        # at the end of each iteration, remove the selected layer, update the self.Gacc
        while self.getGraphSize(GaccTmp) > 0:
            # find the bind layer with smallest end time
            # 找到这些可加速层中计算完成最早的层
            node = self.getSmallEndLayerinGraph(GaccTmp)
            # print(node)
            # 找到该层的加速器前位层和神经网络前位层
            HWpreds = list(self.Gacc.predecessors(node))
            SWpreds = list(self.Gtime.predecessors(node))
            preds = list(set(HWpreds+SWpreds))
            maxend = 0
            # print(1)
            # print(node)
            # print(HWpreds)
            # print(SWpreds)
            # 获得该层的计算的开始是依赖于加速器的计算还是神经网络前一层的计算
            for pred in preds:
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend=self.Gacc.nodes[pred]['end']
            # 如果初次map的时候的时间有误，则重置其开始和结束时间。
            if maxend < self.Gacc.nodes[node]['start']:
                latency = self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['start'] = maxend
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']

            # 开始迭代
            # 此处是因为更新了当前节点，所以我们要更新其后续节点的开始时间
            # then we should start the recursion process, consider the rest node
                HWsuccessor = self.Gacc.successors(node)
                SWsuccessor = self.Gtime.successors(node)
                for HWsu in list(HWsuccessor):
                    self.getNodeShortenBinding(HWsu)
                for SWsu in list(SWsuccessor):
                    self.getNodeShortenBinding(SWsu)

            # remove the selected bind layer
            # 对该层迭代完成后将此层删去，此时该层和(神经网络和加速器)后面的层都被更新过了
            GaccTmp.remove_node(node)

        # 得到所有层中最迟的计算完成时间
        # **************************************************************************************************************
        # 此处是计算ifmofm优化后的加速器终止时间
        nodes = list(self.Gacc)
        for acc in self.AccList:
            accName = acc.getAccName()
            time = 0
            self.Acctimeafterifmofm[accName] = 0
            for node in nodes:
                if self.Gacc.nodes[node]['acc'] == accName:
                    if self.Gacc.nodes[node]['end'] >= time:
                        self.Acctimeafterifmofm[accName] = self.Gacc.nodes[node]['end']
                        time = self.Gacc.nodes[node]['end']

        # 此处不需要通讯时间，因为已经很短了
        nodes = list(self.Gacc)
        complat = {}
        translat = {}
        for acc in self.AccList:
            acc = acc.getAccName()
            complat[acc] = 0
        # 此处获得计算+通讯时间
        for node in nodes:
            accel = self.Gacc.nodes[node]['acc']
            accname = self.getAccName(accel)
            latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
            for acc in self.AccList:
                if accname == acc:
                    tin, tout = acc.getDataInPerf(WEItrans=False, IFMtrans=False, OFMtrans=False)
                    complat[accel] += latency - (tin + tout)


        self.complatafterifmofm = copy.deepcopy(complat)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        self.timeafterifmofm = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))

        # print('after ifm-ofm fusion, the system elapsed time is: ', self.PlatformElapsed)
        # 得到整个系统的能耗
        power = self.getSysPower(self.Gacc)
        self.energyafterifmofm = self.getSysPower(self.Gacc)
        # print('after ifm-ofm fusion, the system energy is: ', power)
        return

    def getModalTune(self, node):
        preds = list(self.GtimeReserve.predecessors(node))
        successs = list(self.GtimeReserve.successors(node))
        for pred in preds:
            # if found we can possibly use same acc for continuous SW layer, try it,
            # 如果当前层的加速器和前位层的加速器不是同一个但层的类型（计算模式）是一样的
            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[pred]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[pred]['attri']['LayerType']:
                # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.getTunePerf(node,pred,'pred')
                # 此处是计算时间比较
                if self.PlatformElapsed > self.getAccMaxElapsed(self.GaccLastTuneTmp):
                    # print(self.getAccMaxElapsed(self.GaccLastTune) , self.getAccMaxElapsed(self.GaccLastTuneTmp))
                    # print('awesome, find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # update the raw mapping
                    self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
                    self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
                    self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)
                    # update the finalized mapping
                    self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
                    self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
                    self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
                    self.AccListLastTune = copy.deepcopy(self.AccListLastTuneTmp) # acclist is newly generated in each iteration
                    # print(self.AccLayersLastTune)
                    # 如果计算完成更快，我们使用现在的映射
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)
        # 如果当前层的加速器和后续层的加速器不是同一个但层的类型（计算模式）是一样的
        # for success in successs:
        #     # if found we can possiblly use same acc for continuous SW layer, try it,
        #     if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[success]['acc'] \
        #             and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[success]['attri']['LayerType']:
        #         # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #         self.getTunePerf(node,success, 'success')
        #         if self.PlatformElapsed  > self.getAccMaxElapsed(self.GaccLastTuneTmp):
        #             # print(self.getAccMaxElapsed(self.GaccLastTune), self.getAccMaxElapsed(self.GaccLastTuneTmp))
        #             # print('awesome, find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #             # update the raw mapping
        #             self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
        #             self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
        #             self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)
        #             # update the finalized mapping
        #             self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
        #             self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
        #             self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
        #             self.AccListLastTune = copy.deepcopy(self.AccListLastTuneTmp)
        #             # print(self.AccLayersLastTune)
        #             self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)

        if successs:
            for success in successs:
                self.getModalTune(success)
        else:
            return

    def getIfmOfmTransLastTune(self):
        # initnode is the starting layer of a acc
        initNode = []
        for node in list(self.GaccLastTuneTmp.nodes):
            if len(list(self.GaccLastTuneTmp.predecessors(node)))==0:
                initNode.append(node)
         # tune each acc's layer time according to modality similarity
        for node in initNode:
            self.getGraphUpdatViaModalLastTune(node)

        # check the first node of Gacc, if its the first node in SW, removed, it can not be shortened
        GaccTmp = copy.deepcopy(self.GaccLastTuneTmp)
        for node in initNode:
            layerOrder = node.split("L")
            if layerOrder[1] == '1':
                GaccTmp.remove_node(node)
            else:
                continue
        # break condition, the GaccTmp does not contain any layer
        # at the end of each iteration, remove the selected layer, update the self.Gacc
        while self.getGraphSize(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallEndLayerinGraph(GaccTmp)
            # print(node)
            HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
            SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))
            preds = HWpreds+SWpreds
            maxend = 0
            for pred in preds:
                if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                    maxend=self.GaccLastTuneTmp.nodes[pred]['end']

            if maxend < self.GaccLastTuneTmp.nodes[node]['start']:
                latency = self.GaccLastTuneTmp.nodes[node]['end']-self.GaccLastTuneTmp.nodes[node]['start']
                self.GaccLastTuneTmp.nodes[node]['start'] = maxend
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + latency
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']

            # then we should start the recursion process, consider the rest node
                HWsuccessor = self.GaccLastTuneTmp.successors(node)
                SWsuccessor = self.GtimeLastTuneTmp.successors(node)
                for HWsu in list(HWsuccessor):
                    self.getNodeShortenBindingLastTune(HWsu)
                for SWsu in list(SWsuccessor):
                    self.getNodeShortenBindingLastTune(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)
        return

    def getdetailedperf(self):
        # 加速器的实际工作时间
        self.accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        self.sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print(accTimeSum, sysCompLat)

    def getColor(self):
        color: int
        color1 = random.randint(16, 255)
        color2 = random.randint(16, 255)
        color3 = random.randint(16, 255)
        color1 = hex(color1)
        color2 = hex(color2)
        color3 = hex(color3)
        ans = "#" + color1[2:] + color2[2:] + color3[2:]
        return ans

    def getperftable(self, i):
        fontsize = 5
        # **************************************************************************************************************
        # 颜色的随机生成
        colormap = {}
        for acc in self.AccList:
            accName = acc.getAccName()
            colormap[accName] = []
            colormap[accName].append(self.getColor())


        plt.figure(i)
        # **************************************************************************************************************
        # 此处是画遍历顺序
        nodes = list(self.Gmap)
        colormapiteration = {}
        colormap_iteration = []
        for i in range(len(self.layers2Map.keys())):
            colormapiteration[i] = []
            colormapiteration[i].append(self.getColor())

        for node in nodes:
            for k, vs in self.layers2Map.items():
                for v in vs:
                    if node == v:
                        colormap_iteration.append(colormapiteration[k])

        ax = plt.subplot(232)
        colormap_iteration = sum(colormap_iteration, [])
        pos = nx.circular_layout(self.Gmap)

        for label in colormapiteration:
            ax.plot(color=colormapiteration[label],label=label)

        for k, vs in self.layers2Map.items():
            nx.draw(self.Gmap, nodelist = vs, with_labels=True, edge_color='r', node_color=colormap_iteration[k], node_size=50,
                    width=1, font_size=fontsize, label=k)

        plt.legend(fontsize=fontsize, fancybox=True, framealpha=0.7)
        plt.title('mapping order for layers', fontsize=10)
        # **************************************************************************************************************
        # 此处是画加速器映射图
        nodes = list(self.Gmap)
        color_map = {}
        # acc_map = []
        colormap = {}
        for acc in self.AccList:
            accName = acc.getAccName()
            colormap[accName] = []
            colormap[accName].append(self.getColor())

        for node in nodes:
            color_map[node] = []
            for key in colormap.keys():
                if self.Gacc.nodes[str(node)]['acc'] == key:
                    color_map[node] = colormap[key]
                    # acc_map.append(key)

        plt.subplot(233)
        pos = nx.circular_layout(self.Gmap)

        for k, vs in self.AccLayers.items():
            # print(vs)
            color = colormap[k]
            nx.draw(self.Gmap, pos, nodelist=vs, with_labels=True, edge_color='r',
                    node_color=color, node_size=50, width=1, font_size=fontsize, label=k)
        plt.title('mapping results', fontsize=10)
        plt.legend(fontsize=fontsize, fancybox=True, framealpha=0.7)
        # **************************************************************************************************************
        # 此处是画每一层的计算时间图(甘特图)
        layertimemap = {}
        nodes = self.Gacc.nodes
        for node in nodes:
            layertimemap[node] = []
            layertimemap[node].append(self.GtimeReserve.nodes[node]['start'])
            layertimemap[node].append(self.GtimeReserve.nodes[node]['end'])
            layertimemap[node].append(self.GtimeReserve.nodes[node]['acc'])

        layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[0][1], x[1][0])))

        plt.subplot(235)
        for k, v in layertimemap.items():
            plt.barh(y=k, width=v[1]-v[0], height=0.5, left=v[0], color = colormap[layertimemap[k][2]])
        plt.yticks(fontsize=5)
        plt.title('detailed time for layers', fontsize=10)
        # **************************************************************************************************************
        # 此处是画加速器的计算时间图
        layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[1][2], x[1][0])))
        plt.subplot(234)
        for k, v in layertimemap.items():
            # layerOrder = k.split("L")
            # print(layerOrder)
            plt.barh(y=v[2], width=v[1]-v[0], height=0.3, left=v[0], color = colormap[layertimemap[k][2]])
        plt.yticks(fontsize=8)  # y轴
        plt.title('operation time for accelerators', fontsize=10)
        # **************************************************************************************************************
        plt.subplot(236)
        value1 = list(self.translataftermapping.values())
        value2 = list(self.complataftermapping.values())
        cell = self.Acctimeaftermapping.keys()
        width = 0.20
        index = np.arange(len(cell))
        plt.xticks(index, cell, fontsize=5)  # 将横坐标用cell替换,fontsize用来调整字体的大小
        plt.bar(index, value2, width, label='computation latency', color='blue')
        plt.bar(index, value1, width, bottom=value2, label='communication latency', color='red')
        # plt.bar(index + width, value2, width, color='blue')

        plt.legend()
# **************************************************************************************************************
        plt.subplot(231)
        pos = nx.planar_layout(self.Amap)
# need to modify the edeg color by hand
        for e in self.Amap.edges():
            self.Amap[e[0]][e[1]]['color'] = 'red'
        self.Amap[acc3.accName][acc1.accName]['color'] = 'grey'
        self.Amap[acc1.accName][acc3.accName]['color'] = 'grey'
        self.Amap[acc3.accName][acc10.accName]['color'] = 'grey'
        self.Amap[acc10.accName][acc3.accName]['color'] = 'grey'

        edge_color_list = [self.Amap[e[0]][e[1]]['color'] for e in self.Amap.edges()]
        nx.draw(self.Amap, pos, with_labels=True, edge_color=edge_color_list, node_color='r', node_size=50,
                width=1, font_size=5)
                # width=[float(v['weight'] / 10) for (r, c, v) in self.Amap.edges(data=True)], font_size=5)
        labels = nx.get_edge_attributes(self.Amap, 'weight')
        nx.draw_networkx_edge_labels(self.Amap, pos, edge_labels=labels)

    def getKnapsackPermu(self, BindLayers= None):
        # we will solve a Knapsack problem problem, select layers in to weight storage
        # we dont change any attri in self.Gtime in self.getMap2AccObjandGtime
        # we only change self.AccList to add assign and bind layer to a accObj
        self.AccList, self.Gtime = self.getMap2AccObjandGtime(self.AccLayers, self.AccList, self.Gtime, BindLayers)
        # just build Gacc graph, does not do any time analysis
        self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        self.GaccReserve= copy.deepcopy(self.Gacc)

    def getBindedTimePermu(self):
        self.getLayers2Graph()
        # lets do a  check to see timegraph and acc graph attribute
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        # self.getHelperGaccTime()
        GaccTmp = copy.deepcopy(self.Gacc)
        # break condition, the GaccTmp does not contain any Binded layer
        # at the end of each iteration, remove the selected bind layer and its successor in GaccTmp, update the self.Gacc
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallBindLayerinGraph(GaccTmp)
            # now update the Gacc graph time, also the Gtime graph time
            layerInfo = self.Gacc.nodes[node]['attri']
            # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
            # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
            acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            acc.getRunPara(layerInfo)
            nodeaccname = acc.getAccName()
            acc.getRunPara(layerInfo)
            # LatSec, Th = acc.getLayerPerf(WEItrans=False)

            predecessors = list(self.Gtime.predecessors(node))
            tDataInfromMem = 0
            tDataOutfromMem = 0
            if predecessors:
                for predecessor in predecessors:
                    predaccname = self.Gtime.nodes[predecessor]['acc']
                    if nodeaccname == predaccname:
                        bandwidth = float('inf')
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=False,
                                                                OFMtrans=False)
                    else:
                        bandwidth = self.Amap[nodeaccname][predaccname]['weight']
                        tDataIn, tDataOut = acc.getDataTranPerf(bandwidth, WEItrans=False, IFMtrans=True,
                                                                OFMtrans=False)
                    tDataInfromMem += tDataIn
                    tDataOutfromMem += tDataOut
                LatSec, Th = acc.getLayerPerf1(WEItrans=False, IFMtrans=False, OFMtrans=False)
                LatSec += (tDataInfromMem + tDataOutfromMem)
            else:
                LatSec, Th = acc.getLayerPerf(WEItrans=False, IFMtrans=True, OFMtrans=False)
            # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
            self.Gtime.nodes[node]['end'] = self.Gtime.nodes[node]['start'] + LatSec
            # print('this layer is shortened to: ', self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start'])
            # then we should start the recursion process, consider the rest node
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBinding(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBinding(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)
            # self.getHelperGaccTime()
        # print('This is time after getting weight binding!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()

        # self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        # print('after binding, the system elapsed time is: ', self.PlatformElapsed)
        # power = self.getSysPower(self.Gacc)
        # print('after binding, the system energy is: ', power)
        #
        # accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        # sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print('after binding, the system CTC time is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        return

    def getIfmOfmTransPermu(self):
        # initnode is the starting layer of a acc
        initNode = []
        for node in list(self.Gacc.nodes):
            if len(list(self.Gacc.predecessors(node))) == 0:
                initNode.append(node)
        # tune each acc's layer time according to modality similarity
        for node in initNode:
            self.getGraphUpdatViaModal(node)
        # 此时缩短了一坨坨相同modal的layer的总体时间 但是graph整体时间还没有去缩短
        # print('This is time after modality similarity!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()

        # check the first node of Gacc, if its the first node in SW, removed, it can not be shortened
        GaccTmp = copy.deepcopy(self.Gacc)
        for node in initNode:
            layerOrder = node.split("L")
            if layerOrder[1] == '1':
                GaccTmp.remove_node(node)
            else:
                continue
        # break condition, the GaccTmp does not contain any layer
        # at the end of each iteration, remove the selected layer, update the self.Gacc
        while self.getGraphSize(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallEndLayerinGraph(GaccTmp)
            # print(node)
            HWpreds = list(self.Gacc.predecessors(node))
            SWpreds = list(self.Gtime.predecessors(node))
            preds = list(set(HWpreds + SWpreds))
            maxend = 0
            for pred in preds:
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend = self.Gacc.nodes[pred]['end']

            if maxend != self.Gacc.nodes[node]['start']:
                latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['start'] = maxend
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']

                # then we should start the recursion process, consider the rest node
                HWsuccessor = self.Gacc.successors(node)
                SWsuccessor = self.Gtime.successors(node)
                for HWsu in list(HWsuccessor):
                    self.getNodeShortenBinding(HWsu)
                for SWsu in list(SWsuccessor):
                    self.getNodeShortenBinding(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)
            # self.getHelperGaccTime()
            # print('1')

        # print('This is time after getting ifm ofm fusion!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # self.getHelperGaccTime()

        # self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        # print('after ifm-ofm fusion, the system elapsed time is: ', self.PlatformElapsed)
        # power = self.getSysPower(self.Gacc)
        # print('after ifm-ofm fusion, the system energy is: ', power)
        return

    def getAccListPermu(self):
        AccCategory = self.getAccCategory(self.AccList)
        for i in range(len(self.LayerOrder)):
            layer = self.LayerOrder[i]
            info = self.Gtime.nodes[layer]['attri']
            if info['LayerType'] == 'CNN':
                self.AccListPermu.append(AccCategory['CNN'])
            elif info['LayerType'] == 'LSTM':
                self.AccListPermu.append(AccCategory['LSTM'])
            elif info['LayerType'] == 'FC':
                self.AccListPermu.append(AccCategory['FC'])

        res = list(itertools.product(*self.AccListPermu))
        for element in res:
            self.Permutations.append(list(element))

    def getMappingOptimal(self, optlat, optenu, iteration):
        sysLatMin = 10000
        sysEnerMin = 10000
        print('The permutation number is: ', len(self.Permutations))
        index = 0
        # iteration = 0

        performanceLog = {'Latency_afterMapping': 0, 'Energy_afterMapping': 0, 'Latency_afterBind': 0,
                          'Energy_afterBind': 0, 'Latency_afterFusion': 0, 'Energy_afterFusion': 0}

        AccLayersBest = {}

        for comb in self.Permutations:
            index = index+1
            iteration += 1
            # print(len(self.Permutations), index, index*100/len(self.Permutations))
            self.Gacc.clear()
            self.AccList = self.AccListNoTouch
            self.getAccLayersInit()
            self.getLayers2Graph()
            self.Gtime = copy.deepcopy(self.Gmap)
            Gmaptmp = copy.deepcopy(self.Gmap)
            AccInitTotLatency = self.getAccInitTotLatency(self.AccList)

            for i in range(len(comb)):
                layerName = self.LayerOrder[i]
                layerAcc = comb[i]
                Accs2MapTmp =[[layerAcc]]
                layers2MapTmp = [(layerName)]

                layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS1(layers2MapTmp, Accs2MapTmp,
                                                                                  AccInitTotLatency,
                                                                                  self.Gmap, self.Gtime,self.Amap, Gmaptmp)

                self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                         self.Gmap, self.Gtime)

            # check if Gmap is empty and if all nodes in Gtime is assigned an acc
            if not self.getGraphValidate(self.Gmap, self.Gtime):
                print('the graph Gmap is not cleaned or Gtime is not built correctly')
                exit()

            Latency_afterMapping = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
            Energy_afterMapping = self.getSysPower(self.Gtime)


            self.getKnapsackPermu()
            self.getBindedTimePermu()

            Latency_afterBind = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
            Energy_afterBind = self.getSysPower(self.Gacc)
            # print('after binding, the system elapsed time is: ', self.PlatformElapsed)
            # print('after binding, the system energy is: ', power)
            self.getIfmOfmTransPermu()
            #
            Latency_afterFusion= copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
            Energy_afterFusion = self.getSysPower(self.Gacc)
            # print('after ifm-ofm fusion, the system elapsed time is: ', self.PlatformElapsed)
            # Energy_afterFusion = self.getSysPower(self.Gacc)
            # print('after ifm-ofm fusion, the system energy is: ', power)

            # print(Latency_afterMapping, Latency_afterBind, Latency_afterFusion)
            if sysLatMin > Latency_afterFusion:
            # if sysLatMin > Latency_afterBind:
                sysLatMin = Latency_afterFusion
                sysEnerMin = Energy_afterFusion
                performanceLog['Latency_afterMapping'] = Latency_afterMapping
                performanceLog['Energy_afterMapping'] = Energy_afterMapping
                performanceLog['Latency_afterBind'] = Latency_afterBind
                performanceLog['Energy_afterBind'] = Energy_afterBind
                performanceLog['Latency_afterFusion'] = Latency_afterFusion
                performanceLog['Energy_afterFusion'] = Energy_afterFusion
                AccLayersBest = copy.deepcopy(self.AccLayers)
                optlat.append(Latency_afterFusion)
                # optenu.append(time.time() - start)
                optenu.append(iteration)


        self.optimallat = copy.deepcopy(sysLatMin)
        print('system minimum latency', sysLatMin, 'system minimum energy', sysEnerMin)
        print(AccLayersBest)
        # print(performanceLog)
        # self.Gacc.clear()
        # self.AccList = self.AccListNoTouch
        # self.getAccLayersInit()
        # self.getLayers2Graph()
        # self.Gtime = copy.deepcopy(self.Gmap)

        return iteration

    def getcomparison(self):
        fontsize = 5
        colormap = {}


if __name__ == "__main__":

    acc1 = cnn_acc_1()
    acc2 = cnn_acc_2()
    acc3 = cnn_acc_3()
    acc4 = cnn_acc_4()
    # acc5 = cnn_acc_5()
    acc6 = cnn_acc_6()
    acc7 = cnn_acc_7()
    acc8 = cnn_acc_8()
    acc9 = cnn_acc_9()
    acc10 = fc_acc_ese()
    acc11 = fc_acc_goDeeper()
    acc12 = lstm_acc_pynq()
    acc13 = lstm_acc_vc707()

    AccsList1 = [acc1, acc2, acc3, acc4, acc6]
    AccsList2 = [acc1, acc2, acc3, acc4]
    AccsList3 = [acc1, acc2, acc3, fc_acc_ese()]
    AccsList4 = [acc1, acc3, fc_acc_ese()]
    AccsList5 = [acc3, acc2, fc_acc_goDeeper()]
    # AccsLists = [AccsList4, AccsList5, AccsList3]
    AccsLists = [AccsList4, AccsList5]
    Modalities = [Modality1]
    # AccsLists = [AccsList4]
    optlat = {}
    optenu = {}
    h2plat = {}
    h2penu = {}


    i = 0
    AccsList = AccsList4
    for Modality in Modalities:
        iteration = 0
        optlat[i] = []
        optenu[i] = []
        h2plat[i] = []
        h2penu[i] = []
        print('Modality case 1')
        print('new accelerator comb')
        print(AccsList)
        print('Optimal')
        NetBand = 0.01
        DDR_WEI = 0.5
        Dtype = 32
        fontsize = 6
        for Acc in AccsList:
            Acc.NetBand = NetBand
            Acc.DDR_WEI = DDR_WEI
            Acc.Dtype = Dtype
        InitMapper = MapperInit()
        InitMapper.NetBand = NetBand
        InitMapper.Dtype = Dtype
        InitMapper.getModalityLayers(Modality)
        # InitMapper.Gmap.add_edge('M2L5', 'M1L4')
        InitMapper.getAccList(AccsList)
        InitMapper.getAccmap(AccsList)
        InitMapper.Amap.add_weighted_edges_from([(acc1.accName, acc3.accName, 5)])
        InitMapper.Amap.add_weighted_edges_from([(acc3.accName, acc10.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc1.accName, acc10.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc3.accName, acc1.accName, 5)])
        InitMapper.Amap.add_weighted_edges_from([(acc10.accName, acc3.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc10.accName, acc1.accName, 0.83)])
        # plt.subplot(231)
        # pos = nx.planar_layout(InitMapper.Amap)
        # nx.draw(InitMapper.Amap, pos, with_labels=True, edge_color='r', node_color='r', node_size=50,
        #         width=[float(v['weight'] / 10) for (r, c, v) in InitMapper.Amap.edges(data=True)], font_size=5)
        # labels = nx.get_edge_attributes(InitMapper.Amap, 'weight')
        # nx.draw_networkx_edge_labels(InitMapper.Amap, pos, edge_labels=labels)
        # plt.show()
        # optimal
        InitMapper.getLayerOrder()
        InitMapper.getAccListPermu()
        start = time.time()
        iteration = InitMapper.getMappingOptimal(optlat[i], optenu[i], iteration)
        end = time.time()
        # optimaltime = end - start
        optlat[i].append(InitMapper.optimallat)

        optenu[i].append(iteration)
        print('the enumerating time is:', end - start, '\n')

        # h2p
        print('H2P')
        for Acc in AccsList:
            Acc.NetBand = NetBand
            Acc.DDR_WEI = DDR_WEI
            Acc.Dtype = Dtype
        InitMapper = MapperInit()
        InitMapper.NetBand = NetBand
        InitMapper.Dtype = Dtype
        InitMapper.getModalityLayers(Modality)
        InitMapper.getAccList(AccsList)
        InitMapper.getAccmap(AccsList)
        InitMapper.Amap.add_weighted_edges_from([(acc1.accName, acc3.accName, 5)])
        InitMapper.Amap.add_weighted_edges_from([(acc3.accName, acc10.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc1.accName, acc10.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc3.accName, acc1.accName, 5)])
        InitMapper.Amap.add_weighted_edges_from([(acc10.accName, acc3.accName, 0.83)])
        InitMapper.Amap.add_weighted_edges_from([(acc10.accName, acc1.accName, 0.83)])
        start = time.time()
        InitMapper.getMappingcom()
        InitMapper.getKnapsack()
        InitMapper.getBindedTime()
        InitMapper.getHomoNeighbor()
        end = time.time()
        print('the H2P time is:', end - start, '\n')
        h2plat[i].append(InitMapper.PlatformElapsed)
        h2penu[i].append(1)
        h2plat[i].append(InitMapper.PlatformElapsed)
        h2penu[i].append(iteration)
        i += 1
        InitMapper.getperftable(i)

    casenum = copy.deepcopy(i)
    # print(optlat, optenu, h2plat, h2penu)
    fig = plt.figure(0)
    # plt.xlim(0, 1)  # 限定横轴的范围
     # 限定纵轴的范围
    for i in range(casenum):
        x = range(i)
        fig.add_subplot(casenum, 1, 1 + i)
        plt.plot(optenu[i], optlat[i], marker='o', label='traversal')
        plt.plot(h2penu[i], h2plat[i], marker='o', label='CHARM')
        plt.xlabel('enumerating times')
        plt.ylabel('system latency')
        plt.legend()
        i += 1
    plt.savefig('0.svg', format='svg', dpi=100)
    plt.show()
