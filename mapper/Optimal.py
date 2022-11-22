import itertools
import sys
import pickle
import time
sys.path.append('C:\\Users\\walker\\Desktop\\FPGA2022\\AcceleratorV2')
sys.path.append('C:\\Users\\walker\\Desktop\\FPGA2022\\Modality')

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
        self.Gmapstart = nx.DiGraph()
        self.Gtime = nx.DiGraph() # a graph to record layers status when mapping
        self.Gtimeifm = nx.DiGraph()
        self.Gtimebind = nx.DiGraph()
        self.Gtimeinibind = nx.DiGraph()
        self.Gtimeiniifm = nx.DiGraph()
        self.Gtimeaftermapping = nx.DiGraph()
        self.GtimeLastTune = nx.DiGraph()
        self.GtimeLastTuneTmp = nx.DiGraph()
        self.Gtimeinimap = nx.DiGraph()
        self.Gacc = nx.DiGraph() # a graph to record acc assigned layer hardware dependency after mapping
        self.GaccLastTune = nx.DiGraph()
        self.GaccLastTuneTmp = nx.DiGraph()
        self.Gtimeini = nx.DiGraph()
        self.commpower = {}
        self.modalities=[] # the different modalities
        self.AccList = [] # the accelerators (sent in as object)
        self.AccListLastTune = []  # the accelerators (sent in as object)
        self.AccListLastTuneTmp = []  # the accelerators (sent in as object)
        self.AccLayers = {} # the assigned layers on accs
        self.AccListini = {}
        self.AccLayersLastTune = {}  # the assigned layers on accs
        self.Acctimeaftermapping = {}
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
        self.cost = {}
        self.costtmp = {}
        self.acclastlayer = {}
        self.PlatformElapsed = float('inf')
        self.NetBand = 0
        self.Dtype = 0
        self.timeaftermapping = 0
        self.timeafterifmofm = 0
        self.energyaftermapping = 0
        self.energyafterbinding = 0
        self.energyafterifmofm = 0
        self.translataftermapping = 0
        self.complataftermapping = 0
        self.complatafterifmofm = 0
        self.syspower = 0
        self.AccListNoTouch = []
        self.GmapPermu = nx.DiGraph()  # a graph that include layer dependency, and task dependency, will be empty after mapping
        self.LayerOrder = []
        self.Permutations = []
        self.AccListPermu = []






    def getSysPower(self, timeGraph):
        allNodes = list(timeGraph.nodes)
        sysPower = 0
        for node in allNodes:
            accName = timeGraph.nodes[node]['acc']
            # Acc = self.getAccbyName(accName, self.AccList)
            # layerInfo = self.Gmap.nodes[node]['attri']
            # Acc.getRunPara(layerInfo)
            start = timeGraph.nodes[node]['start']
            end = timeGraph.nodes[node]['end']
            comm = timeGraph.nodes[node]['tin'] + timeGraph.nodes[node]['tout']
            Acc = self.getAccbyName(accName, self.AccList)
            sysPower = sysPower + (end - start - comm) * Acc.power + (timeGraph.nodes[node]['tin'] + timeGraph.nodes[node]['tout'])* (
                        Acc.NetBand * (10 ** (9-12)
                                       ) * 8) * 3000/ Acc.Dtype

        return sysPower


    # get the acc list and init the acclayer assignment
    def getAccList(self, AccList):
        self.AccList = AccList
        self.AccListNoTouch = AccList
        self.getAccLayersInit()

    def getcommpower(self):
        self.commpower[0]


    #初始化每个加速器上加速的层，都为0
    def getAccLayersInit(self):
        for acc in self.AccList:
            accName = acc.getAccName()
            self.AccLayers[accName]=[]
            self.cost[accName] = 0
            self.acclastlayer[accName] = []

    # get the modalities and create two graph, Gmap to analyze the dependency, Gtime to keep the timeline in analyzing
    def getModalityLayers(self, LayerList):
        #将modalities赋值为layerlist
        self.modalities=LayerList
        self.getLayers2Graph()
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
                return layerInfo

    def getAccmap(self, AccsLists):
        self.Amap = nx.DiGraph()
        p2p = {}
        for AccsList in AccsLists:
            Accs = list(AccsList)
            for Acc in Accs:
                nodename =Acc.getAccName()
                self.Amap.add_node(nodename)
            for Acc in Accs:
                nodename = Acc.getAccName()
                p2p = Acc.getp2p(AccsList)
                for acc in Accs:
                    accname = acc.getAccName()
                    if accname != nodename:
                        self.Amap.add_weighted_edges_from([(nodename, accname, p2p[accname])])
                    # else:
                    #     self.Amap.add_weighted_edges_from([(nodename, accname, float('inf'))])


    def getMapping(self):

        AccInitTotLatency = self.getAccInitTotLatency(self.AccList)
        while self.getGraphSize(self.Gmap) > 0:
            # layers2Map is a dict, key is CNN/LSTM, value is a list of corresponding layers
            # accs2Map is a dict, key is CNN/LSTM, value is a list of corresponding accs2Map

            layers2Map = self.getNoPredesorNode(self.Gmap)
            accs2Map = self.getAccCategory(self.AccList)

            # mapping contains the types of layer to be mapped: CNN/LSTM, or even more
            for type in self.MappingType:
                currentTypeLayers = layers2Map[type]
                if currentTypeLayers:
                    if len(accs2Map[type]) >= len(layers2Map[type]):
                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(layers2Map[type]))
                        layers2MapTmp = layers2Map[type]
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS1(layers2MapTmp, Accs2MapTmp,
                                                                                          AccInitTotLatency,
                                                                                          self.Gmap, self.Gtime)

                        self.AccLayers, self.Gmap, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                                 self.Gmap, self.Gtime)

                    elif len(accs2Map[type]) < len(layers2Map[type]):
                        LayersComb = self.getComb(layers2Map[type], len(accs2Map[type]))

                        Accs2MapTmp = self.getPerRep(accs2Map[type], len(accs2Map[type]))
                        # print(Accs2MapTmp)
                        layers2MapTmp = LayersComb
                        layerTimeMapped, AccInitTotLatency = self.getMinDistanceMappingS2(layers2MapTmp, Accs2MapTmp,
                                                                                          AccInitTotLatency,
                                                                                          self.Gmap, self.Gtime)

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
        # print(self.AccLayers)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        power = self.getSysPower(self.Gtime)
        self.Gtimeinimap = copy.deepcopy(self.Gtime)
        # transLat = self.getJointDNNTrans(self.modalities, self.NetBand, self.Dtype)
        print('after mapping, the system elapsed time is: ', self.PlatformElapsed)
        print('after mapping, the system energy is: ', power)
        # print('after mapping, the system accumulated execution time is: ', accTimeSum)
        # print('after mapping, the system accumulated trans time is: ', transLat)
        return

        # return self.AccLayers, AccInitTotLatency

    def getMappingcom(self,Modality):
        # 有Gmap和Gtime
        InitMapper.getModalityLayers(Modality)
        AccInitTotLatency = self.getAccInitTotLatency(self.AccList)
        Gmaptmp = copy.deepcopy(self.Gmap)
        Gmapstart = copy.deepcopy(self.Gmap)
        self.Gtime  = copy.deepcopy(self.Gmap)


        accs2Map = self.getAccCategory(self.AccList)

        self.Gtime = copy.deepcopy(self.Gmap)
        Gmapstarttmp = copy.deepcopy(Gmapstart)

        iteration = 0
        i = 0
        layers2Map = {}
        while self.getGraphSize(Gmaptmp) > 0:
            layers2Map[i] = []
            for type in self.MappingType:
                starters = self.getNoPredesorNode(Gmaptmp)
                currentTypeLayers = starters[type]
                if currentTypeLayers:
                    distance = float('inf')
# **********************************************************************************************************************
#                     edition2
                    layer2acc = self.getPerRep(accs2Map[type], min(len(accs2Map[type]), len(starters[type])))
                    startersTmp = self.getComb(starters[type], min(len(accs2Map[type]), len(starters[type])))
#                     layer2acc = self.getPerRep(accs2Map[type], len(starters[type]))
#                     startersTmp = self.getComb(starters[type], len(starters[type]))
                    layers2Maptmp = []
# **********************************************************************************************************************
#                     edition2
                    layerTimeMapped, AccInitTotLatency, layers2Maptmp, iteration, self.cost, self.acclastlayer = self.getMinLatency(startersTmp, layer2acc,
                                                                                           AccInitTotLatency,
                                                                                           Gmaptmp, self.Gtime, iteration, self.cost, self.acclastlayer)
                    layers2Map[i] = layers2Maptmp

                    self.AccLayers, Gmaptmp, self.Gtime = self.GraphChange(layerTimeMapped, self.AccLayers,
                                                                             Gmaptmp, self.Gtime)

                else:
                    continue

                i += 1

        # print(layers2Map)

        self.Gtimeaftermapping = copy.deepcopy(self.Gtime)
        self.Acctimeaftermapping = copy.deepcopy(AccInitTotLatency)

        self.layers2Map = copy.deepcopy(layers2Map)

        if not self.getGraphValidate(Gmaptmp, self.Gtime):
            print('the graph Gmap is not cleaned or Gtime is not built correctly')
            exit()

        self.cost = {}
        self.acclastlayer = {}
        self.GtimeReserve = copy.deepcopy(self.Gtime)
        self.AccLayersReserve = copy.deepcopy(self.AccLayers)
        self.AccListReserve = copy.deepcopy(self.AccList)
        self.Gtimeini = copy.deepcopy(self.Gtime)
        self.AccListini = copy.deepcopy(self.AccLayers)
        print('accelerator allocation:')
        print(self.AccLayers, '\n')
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))
        self.timeaftermapping = copy.deepcopy(self.getAccMaxElapsed(self.Gtime))

        # self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        # sysCompLat = self.getJointDNNCompLat(self.AccList, self.GaccLastTune)
        power = self.getSysPower(self.Gtime)
        self.energyaftermapping = self.getSysPower(self.Gtime)
        self.Gtimeinimap = copy.deepcopy(self.Gtime)
        print('after mapping, the system elapsed time is: ', self.PlatformElapsed)
        print('after mapping, the system energy is: ', power)

        return

    def getLayerOrder(self):
        self.GmapPermu = copy.deepcopy(self.Gmap)
        while self.getGraphSize(self.GmapPermu) > 0:
            layers2Map = self.getNoPredesorNode(self.GmapPermu)
            # mapping contains the types of layer to be mapped: CNN/LSTM, or even more
            avaLayers = []
            #mappingtype包括CNN，LSTM，FCN
            for type in self.MappingType:
                currentTypeLayers = layers2Map[type]
                avaLayers = avaLayers + currentTypeLayers
            self.LayerOrder.append(avaLayers[0])
            self.GmapPermu.remove_node(avaLayers[0])


    def getNodeShortenBinding(self, node):
        HWpreds = list(self.Gacc.predecessors(node))
        SWpreds = list(self.Gtime.predecessors(node))

        preds = HWpreds + SWpreds

        maxend = 0
        for pred in preds:
            if self.Gacc.nodes[pred]['end'] > maxend:
                maxend = self.Gacc.nodes[pred]['end']

        if maxend < self.Gacc.nodes[node]['start']:
            latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
            self.Gacc.nodes[node]['start'] = maxend
            self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + latency
            self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
            self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
            HWsuccessor = self.Gacc.successors(node)
            SWsuccessor = self.Gtime.successors(node)
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

        if maxend < self.GaccLastTuneTmp.nodes[node]['start']:
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

    def getBindedTime(self):
        # lets do a  check to see timegraph and acc graph attribute
        self.getLayerBindVarify(self.Gtime, self.Gacc)
        # self.getHelperGaccTime()
        GaccTmp=copy.deepcopy(self.Gacc)
        # break condition, the GaccTmp does not contain any Binded layer
        # at the end of each iteration, remove the selected bind layer and its successor in GaccTmp, update the self.Gacc
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            # find the bind layer with smallest end time
            node = self.getSmallBindLayerinGraph(GaccTmp)
            # now update the Gacc graph time, also the Gtime graph time
            # **********************************************************************
            # layerInfo = self.Gacc.nodes[node]['attri']
            # # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
            # # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
            # acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
            # nodeaccname = acc.getAccName()
            # acc.getRunPara(layerInfo)
            # LatSec, Th = acc.getLayerPerf(WEItrans=False,IFMtrans=True,OFMtrans=True)
            #
            # self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start']+ LatSec
            # self.Gtime.nodes[node]['end'] = self.Gtime.nodes[node]['start'] + LatSec
            # **********************************************************************
            HWpreds = list(self.Gacc.predecessors(node))
            SWpreds = list(self.Gtime.predecessors(node))
            preds = HWpreds + SWpreds
            maxend = 0
            # print('****************************')
            # print(node)
            for pred in preds:
                # print(pred, self.GaccLastTuneTmp.nodes[pred]['end'])
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend = self.Gacc.nodes[pred]['end']
            #     print(maxend)
            # print(self.GaccLastTuneTmp.nodes[node]['start'])
            if maxend < self.Gacc.nodes[node]['start']:
                # HWsuccessors = self.Gacc.successors(node)
                # SWsuccessors = self.Gtime.successors(node)

                # latency = self.GaccLastTuneTmp.nodes[node]['end'] - self.GaccLastTuneTmp.nodes[node]['start']
                layerInfo = self.Gacc.nodes[node]['attri']
                # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
                # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                acc.getRunPara(layerInfo)
                LatSec, tin, tout = acc.getLayerPerf(WEItrans=False)
                self.Gacc.nodes[node]['start'] = maxend
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']

            else:
                layerInfo = self.Gacc.nodes[node]['attri']
                # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
                # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
                acc = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                acc.getRunPara(layerInfo)
                LatSec, tin, tout = acc.getLayerPerf(WEItrans=False)
                # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
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

        self.Gtimeaftermapping = copy.deepcopy(self.Gtime)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        print('after binding, the system elapsed time is: ', self.PlatformElapsed)
        power = self.getSysPower(self.Gtime)
        self.energyafterbinding = power
        print('after binding, the system energy is: ', power)
        self.Gtimeinibind = copy.deepcopy(self.Gtime)
        #
        # accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        # sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print('after binding, the system CTC time is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        return



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
        while self.getBindLayerinGraphNum(GaccTmp) > 0:
            node = self.getSmallBindLayerinGraph(GaccTmp)
            # now update the Gacc graph time, also the Gtime graph time
            HWpreds = list(self.GaccLastTuneTmp.predecessors(node))
            SWpreds = list(self.GtimeLastTuneTmp.predecessors(node))
            preds = HWpreds + SWpreds
            maxend = 0
            # print('****************************')
            # print(node)
            for pred in preds:
                # print(pred, self.GaccLastTuneTmp.nodes[pred]['end'])
                if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                    maxend = self.GaccLastTuneTmp.nodes[pred]['end']
            # print(maxend)
            # print(self.GaccLastTuneTmp.nodes[node]['start'])
            if maxend != self.GaccLastTuneTmp.nodes[node]['start']:
                # HWsuccessors = self.GaccLastTuneTmp.successors(node)
                # SWsuccessors = self.GtimeLastTuneTmp.successors(node)

                # latency = self.GaccLastTuneTmp.nodes[node]['end'] - self.GaccLastTuneTmp.nodes[node]['start']
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
                # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                acc.getRunPara(layerInfo)
                LatSec, tin, tout = acc.getLayerPerf(WEItrans=False)
                self.GaccLastTuneTmp.nodes[node]['start'] = maxend
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
            else:
                layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                # we will use Gacc graph to do this, the bind layer end time can be reduced for sure
                # acc = self.getAccObjbyName(self.Gacc.nodes[node]['accName'])
                acc = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                acc.getRunPara(layerInfo)
                LatSec, tin, tout = acc.getLayerPerf(WEItrans=False)
                # print('this layer is originally: ', self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start'])
                self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                self.GtimeLastTuneTmp.nodes[node]['end'] = self.GtimeLastTuneTmp.nodes[node]['start'] + LatSec
                self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
            # then we should start the recursion process, consider the rest node

            HWsuccessor = self.GaccLastTuneTmp.successors(node)
            SWsuccessor = self.GtimeLastTuneTmp.successors(node)
            for HWsu in list(HWsuccessor):
                self.getNodeShortenBindingLastTune(HWsu)
            for SWsu in list(SWsuccessor):
                self.getNodeShortenBindingLastTune(SWsu)

            # remove the selected bind layer
            GaccTmp.remove_node(node)

        self.Gtimebind = copy.deepcopy(self.GtimeLastTuneTmp)
        return

    def getGraphUpdatViaModal(self, node):

        preds = list(self.Gacc.predecessors(node))
        successs = list(self.Gacc.successors(node))

        #special case init node and last node, or the node is init and last node
        #the node is init and last node, return, no change should be done
        if not preds and not successs:
            return
        #the node is init, it can has the same modal with its successor or not
        if not preds and successs:
            success = successs[0]
            # the node is init, it can has the same modal with its successor
            if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['toutx'] = 0
                self.Gacc.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['toutx'] = 0

                self.getGraphUpdatViaModal(success)

            # X (red) red, the two red not continuous
            elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModal(success)
            # the node is init, it does not has the same modal with its successor
            # X (red) green
            elif not self.getModalCheck(node, success):
                # do nothing, recursion, the next node
                self.getGraphUpdatViaModal(success)

        #the node is last node

        if preds and not successs:
            pred=preds[0]
            # the node is last, it can has the same modal with its pred
            if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gacc.nodes[node]['tinx'] = 0
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tinx'] = 0
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
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gacc.nodes[node]['tinx'] = 0
                self.Gacc.nodes[node]['toutx'] = 0
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tinx'] = 0
                self.Gtime.nodes[node]['toutx'] = 0
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case 1-1: red (red) red, first two are continuous, last two not
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success) and self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gacc.nodes[node]['tinx'] = 0
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tinx'] = 0
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case 1-2: red (red) red, first two are not continuous, last two are
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gacc.nodes[node]['toutx'] = 0
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['toutx'] = 0
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case 1-3: red (red) red, they are not continuous
            elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case: green (red) red
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    self.getDependCheck(node, success):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gacc.nodes[node]['toutx'] = 0
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['toutx'] = 0

                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case: green (red) red, but two red are not continuous
            elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                    not self.getDependCheck(node, success):
                # dont need to change anything for this node, the weight trans is already tuned
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

            # case: red (red) green
            elif not self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                 self.getDependCheck(pred, node):
                accObj = self.getAccObjbyName(self.Gacc.nodes[node]['acc'], self.AccList)
                layerInfo = self.Gacc.nodes[node]['attri']
                isBind = self.Gacc.nodes[node]['bind']
                if isBind:
                    WEItrans = False
                else:
                    WEItrans = True
                accObj.getRunPara(layerInfo)
                LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                self.Gacc.nodes[node]['start'] = self.Gacc.nodes[pred]['end']
                self.Gtime.nodes[node]['start'] = self.Gacc.nodes[node]['start']
                self.Gacc.nodes[node]['end'] = self.Gacc.nodes[node]['start'] + LatSec
                self.Gtime.nodes[node]['end'] = self.Gacc.nodes[node]['end']
                self.Gacc.nodes[node]['tin'] = tin
                self.Gacc.nodes[node]['tout'] = tout
                self.Gtime.nodes[node]['tin'] = tin
                self.Gtime.nodes[node]['tout'] = tout
                # recursion, the next node
                self.getGraphUpdatViaModal(success)

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
        initNode = []
        for node in list(self.Gacc.nodes):
            if len(list(self.Gacc.predecessors(node)))==0:
                initNode.append(node)

        for node in initNode:
            self.getGraphUpdatViaModal(node)

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
            preds = list(set(HWpreds+SWpreds))
            maxend = 0

            for pred in preds:
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend=self.Gacc.nodes[pred]['end']
            if maxend < self.Gacc.nodes[node]['start']:
                latency = self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start']
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


        # **************************************************************************************************************
        # nodes = list(self.Gacc)
        # for acc in self.AccList:
        #     accName = acc.getAccName()
        #     time = 0
        #     self.Acctimeafterifmofm[accName] = 0
        #     for node in nodes:
        #         if self.Gacc.nodes[node]['acc'] == accName:
        #             if self.Gacc.nodes[node]['end'] >= time:
        #                 self.Acctimeafterifmofm[accName] = self.Gacc.nodes[node]['end']
        #                 time = self.Gacc.nodes[node]['end']

        # nodes = list(self.Gacc)
        # complat = {}
        # translat = {}
        # for acc in self.AccList:
        #     acc = acc.getAccName()
        #     complat[acc] = 0
        # for node in nodes:
        #     accel = self.Gacc.nodes[node]['acc']
        #     accname = self.getAccName(accel)
        #     latency = self.Gacc.nodes[node]['end'] - self.Gacc.nodes[node]['start']
        #     for acc in self.AccList:
        #         if accname == acc:
        #             tin, tout = acc.getDataInPerf(WEItrans=False, IFMtrans=False, OFMtrans=False)
        #             complat[accel] += latency - (tin + tout)
        #
        #
        # self.complatafterifmofm = copy.deepcopy(complat)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        self.timeafterifmofm = copy.deepcopy(self.getAccMaxElapsed(self.Gacc))
        self.Gtimeiniifm = copy.deepcopy(self.Gtime)
        print('after ifm-ofm fusion, the system elapsed time is: ', self.PlatformElapsed)
        # 得到整个系统的能耗
        self.syspower = self.getSysPower(self.Gacc)
        print('after ifm-ofm fusion, the system energy is: ', self.syspower)
        return

    def getModalTune(self, node):
        preds = list(self.GtimeReserve.predecessors(node))
        successs = list(self.GtimeReserve.successors(node))
        # preds = list(self.GtimeLastTune.predecessors(node))
        # successs = list(self.GtimeLastTune.successors(node))
        for pred in preds:
            # if found we can possibly use same acc for continuous SW layer, try it,
            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[pred]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[pred]['attri']['LayerType']:
                # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.getTunePerf(node,pred,'pred')
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
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)
                elif self.PlatformElapsed == self.getAccMaxElapsed(self.GaccLastTuneTmp) and self.syspower > self.getSysPower(self.GaccLastTuneTmp):
                    self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
                    self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
                    self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)
                    # update the finalized mapping
                    self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
                    self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
                    self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
                    self.AccListLastTune = copy.deepcopy(
                        self.AccListLastTuneTmp)  # acclist is newly generated in each iteration
                    # print(self.AccLayersLastTune)
                    self.syspower = self.getSysPower(self.GaccLastTuneTmp)
                # *****************************************
        # for success in successs:
        #     # if found we can possiblly use same acc for continuous SW layer, try it,
        #     if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[success]['acc'] \
        #             and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[success]['attri']['LayerType']:
        #         # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #         self.getTunePerf(node, success, 'success')
        #         if self.PlatformElapsed > self.getAccMaxElapsed(self.GaccLastTuneTmp):
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

    def getModalTuneh2h(self, node):
        preds = list(self.GtimeReserve.predecessors(node))
        successs = list(self.GtimeReserve.successors(node))
        # preds = list(self.GtimeLastTune.predecessors(node))
        # successs = list(self.GtimeLastTune.successors(node))
        for pred in preds:
            # if found we can possibly use same acc for continuous SW layer, try it,
            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[pred]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[pred]['attri']['LayerType']:
                # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.getTunePerf(node,pred,'pred')
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
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)

                # *****************************************
        for success in successs:
            # if found we can possiblly use same acc for continuous SW layer, try it,
            if self.GtimeReserve.nodes[node]['acc'] != self.GtimeReserve.nodes[success]['acc'] \
                    and self.GtimeReserve.nodes[node]['attri']['LayerType'] == self.GtimeReserve.nodes[success]['attri']['LayerType']:
                # print('Try to find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.getTunePerf(node, success, 'success')
                if self.PlatformElapsed > self.getAccMaxElapsed(self.GaccLastTuneTmp):
                    # print(self.getAccMaxElapsed(self.GaccLastTune), self.getAccMaxElapsed(self.GaccLastTuneTmp))
                    # print('awesome, find a layer that can be swapped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # update the raw mapping
                    self.GaccReserve = copy.deepcopy(self.GaccReserveTmp)
                    self.GtimeReserve = copy.deepcopy(self.GtimeReserveTmp)
                    self.AccLayersReserve = copy.deepcopy(self.AccLayersReserveTmp)
                    # update the finalized mapping
                    self.GaccLastTune = copy.deepcopy(self.GaccLastTuneTmp)
                    self.GtimeLastTune = copy.deepcopy(self.GtimeLastTuneTmp)
                    self.AccLayersLastTune = copy.deepcopy(self.AccLayersLastTuneTmp)
                    self.AccListLastTune = copy.deepcopy(self.AccListLastTuneTmp)
                    # print(self.AccLayersLastTune)
                    self.PlatformElapsed = self.getAccMaxElapsed(self.GaccLastTuneTmp)


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
            # print('****************************')
            # print(node)
            for pred in preds:
                # print(pred, self.GaccLastTuneTmp.nodes[pred]['end'])
                if self.GaccLastTuneTmp.nodes[pred]['end'] > maxend:
                    maxend=self.GaccLastTuneTmp.nodes[pred]['end']
            #     print(maxend)
            # print(self.GaccLastTuneTmp.nodes[node]['start'])
            if maxend < self.GaccLastTuneTmp.nodes[node]['start']:
                HWsuccessors = self.GaccLastTuneTmp.successors(node)
                SWsuccessors = self.GtimeLastTuneTmp.successors(node)
                # # *******************************************************
                # latency = 0
                # nodeaccname = self.GtimeLastTuneTmp.nodes[node]['acc']
                # nodeacc = self.getAccObjbyName(nodeaccname, self.AccList)
                # layerInfo = self.Gmap.nodes[node]['attri']
                # isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                # if isBind:
                #     WEItrans = False
                # else:
                #     WEItrans = True
                # nodeacc.getRunPara(layerInfo)
                # LatSec, Th = nodeacc.getLayerPerf1()
                # latency += LatSec
                # for SWsuccessor in SWsuccessor:
                #     if SWsuccessor == HWsuccessors:
                #         tDataIn, tDataOut = acc.getDataInPerf(WEItrans=isBind, IFMtrans=True,
                #                                               OFMtrans=False)
                #         latency +=
                #

                # *******************************************************
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

            # print(self.GaccLastTuneTmp.nodes[node]['start'])
            # remove the selected bind layer
            GaccTmp.remove_node(node)
        self.Gtimeifm = copy.deepcopy(self.GtimeLastTuneTmp)
        return

    def getdetailedperf(self):
        # 加速器的实际工作时间
        self.accTimeSum = self.getAccsAccuTimeSum(self.Gacc)
        self.sysCompLat = self.getJointDNNCompLat(self.AccList, self.Gacc)
        # print(accTimeSum, sysCompLat)

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

    def getMappingOptimal(self, start, optlat, optenu):
        sysLatMin = 10000
        sysEnerMin = 10000
        print('The permutation number is: ', len(self.Permutations))
        index = 0

        performanceLog = {'Latency_afterMapping': 0, 'Energy_afterMapping': 0, 'Latency_afterBind': 0,
                          'Energy_afterBind': 0, 'Latency_afterFusion': 0, 'Energy_afterFusion': 0}

        AccLayersBest = {}

        for comb in self.Permutations:
            index = index+1
            # print(len(self.Permutations), index, index*100/len(self.Permutations))
            self.Gacc.clear()
            self.AccList = self.AccListNoTouch
            self.getAccLayersInit()
            self.getLayers2Graph()
            self.Gtime = copy.deepcopy(self.Gmap)
            # Gmaptmp = copy.deepcopy(self.Gmap)
            AccInitTotLatency = self.getAccInitTotLatency(self.AccList)

            for i in range(len(comb)):
                layerName = self.LayerOrder[i]
                layerAcc = comb[i]
                Accs2MapTmp =[[layerAcc]]
                layers2MapTmp = [(layerName)]

                layerTimeMapped, AccInitTotLatency = self.getMinDistanceMapping(layers2MapTmp, Accs2MapTmp,
                                                                                  AccInitTotLatency,
                                                                                  self.Gmap, self.Gtime)

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
            # print(comb)
            # print(Latency_afterMapping, Latency_afterBind, Latency_afterFusion)
            # print(Latency_afterFusion)
            # if sysLatMin > Latency_afterFusion:
            if sysLatMin > Latency_afterFusion:
                sysLatMin = Latency_afterFusion
                sysEnerMin = Energy_afterFusion
                performanceLog['Latency_afterMapping'] = Latency_afterMapping
                performanceLog['Energy_afterMapping'] = Energy_afterMapping
                performanceLog['Latency_afterBind'] = Latency_afterBind
                performanceLog['Energy_afterBind'] = Energy_afterBind
                performanceLog['Latency_afterFusion'] = Latency_afterFusion
                performanceLog['Energy_afterFusion'] = Energy_afterFusion
                AccLayersBest = copy.deepcopy(self.AccLayers)
                optlat.append(Latency_afterBind)
                optenu.append(time.time() - start)


        self.optimallat = copy.deepcopy(sysLatMin)
        print('system minimum latency', sysLatMin, 'system minimum energy', sysEnerMin)
        print(AccLayersBest)
        # print(performanceLog)
        self.Gacc.clear()
        self.AccList = self.AccListNoTouch
        self.getAccLayersInit()
        self.getLayers2Graph()
        self.Gtime = copy.deepcopy(self.Gmap)

        return

    def getIfmOfmTransPermu(self):
        # initnode is the starting layer of a acc
        initNode = []
        for node in list(self.Gacc.nodes):
            if len(list(self.Gacc.predecessors(node)))==0:
                initNode.append(node)
         # tune each acc's layer time according to modality similarity
        for node in initNode:
            self.getGraphUpdatViaModal(node)
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
            preds = list(set(HWpreds+SWpreds))
            maxend = 0
            for pred in preds:
                if self.Gacc.nodes[pred]['end'] > maxend:
                    maxend=self.Gacc.nodes[pred]['end']

            if maxend < self.Gacc.nodes[node]['start']:
                latency = self.Gacc.nodes[node]['end']-self.Gacc.nodes[node]['start']
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

    def getKnapsackPermu(self, BindLayers=None):
        # we will solve a Knapsack problem problem, select layers in to weight storage
        # we dont change any attri in self.Gtime in self.getMap2AccObjandGtime
        # we only change self.AccList to add assign and bind layer to a accObj
        self.AccList, self.Gtime = self.getMap2AccObjandGtime(self.AccLayers, self.AccList, self.Gtime, BindLayers)
        # just build Gacc graph, does not do any time analysis
        self.Gacc = self.getAccTimeGraph(self.AccList, self.Gtime, self.Gacc)
        self.GaccReserve = copy.deepcopy(self.Gacc)

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
            LatSec, tin, tout = acc.getLayerPerf(WEItrans=False)

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

    def getHomoNeighbor(self):
        # self.GtimeLastTune, self.AccLayersLastTune from getMapping
        # self.GaccLastTune from getBindedTime
        # they are result for self.getMapping
        lastNode = []
        for node in list(self.Gtime.nodes):
            if len(list(self.Gtime.successors(node))) == 0:
                lastNode.append(node)
        lastNode = sorted(lastNode, key=lambda x: self.Gtime.nodes[x]['end'], reverse=True)
        # print('******************************************************************************************')
        # print(lastNode)
        initNode = copy.deepcopy(lastNode)
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

        start = time.time()
        for node in initNodeRepeat:
            self.getModalTune(node)
        end = time.time()
        timeElapse = end - start

        accTimeSum = self.getAccsAccuTimeSum(self.GaccLastTune)
        sysCompLat = self.getJointDNNCompLat(self.AccList, self.GaccLastTune)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.GaccLastTune))
        # print(self.GaccLastTune)
        print(self.AccLayersLastTune)
        print('after color flip, the system elapsed time is: ', self.PlatformElapsed)
        print('after color flip, the system CTC is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        power = self.getSysPower(self.GaccLastTune)
        print('after color flip, the system energy is: ', power)
        print('the color flip search time is : ', timeElapse)

        # print('time check')
        # nodes = list(self.GtimeLastTune.nodes())
        # for node in nodes:
        #     print(self.GtimeLastTune.nodes[node]['bind'])

        print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def getHomoNeighborh2h(self):
        # self.GtimeLastTune, self.AccLayersLastTune from getMapping
        # self.GaccLastTune from getBindedTime
        # they are result for self.getMapping
        lastNode = []
        for node in list(self.Gtime.nodes):
            if len(list(self.Gtime.successors(node))) == 0:
                lastNode.append(node)
        lastNode = sorted(lastNode, key=lambda x: self.Gtime.nodes[x]['end'], reverse=True)
        # print('******************************************************************************************')
        # print(lastNode)
        initNode = copy.deepcopy(lastNode)
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

        start = time.time()
        for node in initNodeRepeat:
            self.getModalTuneh2h(node)
        end = time.time()
        timeElapse = end - start

        accTimeSum = self.getAccsAccuTimeSum(self.GaccLastTune)
        sysCompLat = self.getJointDNNCompLat(self.AccList, self.GaccLastTune)
        self.PlatformElapsed = copy.deepcopy(self.getAccMaxElapsed(self.GaccLastTune))
        # print(self.GaccLastTune)
        print(self.AccLayersLastTune)
        print('after color flip, the system elapsed time is: ', self.PlatformElapsed)
        print('after color flip, the system CTC is: ', sysCompLat, accTimeSum, sysCompLat / accTimeSum)
        power = self.getSysPower(self.GaccLastTune)
        print('after color flip, the system energy is: ', power)
        print('the color flip search time is : ', timeElapse)

        # print('time check')
        # nodes = list(self.GtimeLastTune.nodes())
        # for node in nodes:
        #     print(self.GtimeLastTune.nodes[node]['bind'])

        print('Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def getTunePerf(self, node, neighbor, relation):
        # we need to update both self.GtimeLastTune, self.GaccLastTune, self.AccLayersLastTune
        # a node change acc assignment to its pred
        self.GtimeLastTuneTmp = copy.deepcopy(self.GtimeReserve)
        self.GaccLastTuneTmp = copy.deepcopy(self.GaccReserve)
        self.AccLayersLastTuneTmp = copy.deepcopy(self.AccLayersReserve)
        if relation == 'pred':
            nodeAcc = self.GtimeLastTuneTmp.nodes[node]['acc']
            predAcc = self.GtimeLastTuneTmp.nodes[neighbor]['acc']
            # change the assignment attribute
            self.GtimeLastTuneTmp.nodes[node]['acc'] = predAcc
            self.GaccLastTuneTmp.nodes[node]['acc'] = predAcc
            self.AccLayersLastTuneTmp[nodeAcc].remove(node)
            predAccPos = self.AccLayersLastTuneTmp[predAcc].index(neighbor) + 1
            self.AccLayersLastTuneTmp[predAcc].insert(predAccPos, node)
            # change the connections in self.GtimeLastTune, self.GaccLastTune
            # now we need to update self.GtimeLastTune, self.GaccLastTune since we insert and remove a layer in two acc respectively
            # we dont need to change self.GtimeLastTune's edge
            # find pred's successor in self.GaccLastTune, unlink the pred and successor
            predSuccOnAcc = list(self.GaccLastTuneTmp.successors(neighbor))
            nodesuccOnAcc = list(self.GaccLastTuneTmp.successors(node))
            nodepredOnAcc = list(self.GaccLastTuneTmp.predecessors(node))
            if predSuccOnAcc:
                predSuccOnAcc = predSuccOnAcc[0]
                self.GaccLastTuneTmp.remove_edge(neighbor, predSuccOnAcc)
                self.GaccLastTuneTmp.add_edge(neighbor, node)
                self.GaccLastTuneTmp.add_edge(node, predSuccOnAcc)
            elif not predSuccOnAcc:
                self.GaccLastTuneTmp.add_edge(neighbor, node)
                # find node's successor and predecessor in self.GaccLastTune, link the pred and successor
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

        # dont do anything to self.AccList. every time we use a clean one. after a iteration, we
        # get a self.Acclist with arribute defined
        self.GaccReserveTmp = self.GaccLastTuneTmp
        self.GtimeReserveTmp = self.GtimeLastTuneTmp
        self.AccLayersReserveTmp = self.AccLayersLastTuneTmp

        # self.AccListLastTuneTmp, self.GtimeLastTuneTmp = self.getMap2AccObjandGtime(self.AccLayersLastTuneTmp,self.AccListLastTuneTmp,self.GtimeLastTuneTmp, None)
        self.getBindedTimeLastTune()
        self.getIfmOfmTransLastTune()


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
        # preds = list(self.GtimeLastTuneTmp.predecessors(node))
        # successs = list(self.GtimeLastTuneTmp.successors(node))
        preds = list(self.GaccLastTuneTmp.predecessors(node))
        successs = list(self.GaccLastTuneTmp.successors(node))
        #special case init node and last node, or the node is init and last node
        #the node is init and last node, return, no change should be done
        if not preds and not successs:
            return
        #the node is init, it can has the same modal with its successor or not
        if not preds and successs:
            for success in successs:
            # success = successs[0]
            # the node is init, it can has the same modal with its successor
            # X (red) red
                if self.getModalCheck(node,success) and self.getDependCheck(node,success):
                    accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                    layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                    isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                    if isBind:
                        WEItrans = False
                    else:
                        WEItrans = True
                    accObj.getRunPara(layerInfo)
                    LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                    self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                    self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                    self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                    self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                    self.GaccLastTuneTmp.nodes[node]['toutx'] = 0
                    self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                    self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                    self.GtimeLastTuneTmp.nodes[node]['toutx'] = 0
                    # recursion, the next node
                    self.getGraphUpdatViaModalLastTune(success)

                else:
                    self.getGraphUpdatViaModalLastTune(success)
                # # X (red) red, the two red not continuous
                # elif self.getModalCheck(node,success) and not self.getDependCheck(node,success):
                #     # do nothing, recursion, the next node
                #     self.getGraphUpdatViaModalLastTune(success)
                # # the node is init, it does not has the same modal with its successor
                # # X (red) green
                # elif not self.getModalCheck(node, success):
                #     # do nothing, recursion, the next node
                #     self.getGraphUpdatViaModalLastTune(success)

        #the node is last node
        if preds and not successs:
            # pred=preds[0]
            # the node is last, it can has the same modal with its pred
            # red (red) X
            for pred in preds:
                if self.getModalCheck(pred, node) and self.getDependCheck(pred, node):
                    accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                    layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                    isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                    if isBind:
                        WEItrans = False
                    else:
                        WEItrans = True
                    accObj.getRunPara(layerInfo)
                    LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                    self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                    self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                    self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                    self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                    self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                    self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                    self.GaccLastTuneTmp.nodes[node]['tinx'] = 0
                    self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                    self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                    self.GtimeLastTuneTmp.nodes[node]['tinx'] = 0
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
            for pred in preds:
                for success in successs:
            # (layer) is our target
            # case 1-0: red (red) red
                    if self.getModalCheck(node,success) and self.getModalCheck(pred,node) and \
                            self.getDependCheck(node,success) and self.getDependCheck(pred,node):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=False)
                        self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                        self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GaccLastTuneTmp.nodes[node]['toutx'] = 0
                        self.GaccLastTuneTmp.nodes[node]['tinx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['toutx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tinx'] = 0
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case 1-1: red (red) red, first two are continuous, last two not
                    elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                            self.getDependCheck(pred, node) and not self.getDependCheck(node, success):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                        self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                        self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GaccLastTuneTmp.nodes[node]['tinx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['tinx'] = 0
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case 1-2: red (red) red, first two are not continuous, last two are
                    elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                         self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GaccLastTuneTmp.nodes[node]['toutx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['toutx'] = 0
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case 1-3: red (red) red, they are not continuous
                    elif self.getModalCheck(node, success) and self.getModalCheck(pred, node) and \
                         not self.getDependCheck(node, success) and not self.getDependCheck(pred, node):
                        # dont need to change anything for this node, the weight trans is already tuned
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)


                    # case: green (red) red
                    elif self.getModalCheck(node, success) and self.getDependCheck(node, success) and \
                            not self.getModalCheck(pred, node):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=False, IFMtrans=True)
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GaccLastTuneTmp.nodes[node]['toutx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['toutx'] = 0

                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case: green (red) red, but two red are not continuous
                    elif self.getModalCheck(node, success) and not self.getModalCheck(pred, node) and \
                            not self.getDependCheck(node, success):
                        # dont need to change anything for this node, the weight trans is already tuned
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=True)
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        # recursion, the next node
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case: red (red) green
                    elif self.getModalCheck(pred, node) and \
                         self.getDependCheck(pred, node) and not self.getModalCheck(node, success):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'],self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=False)
                        self.GaccLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[pred]['end']
                        self.GtimeLastTuneTmp.nodes[node]['start'] = self.GaccLastTuneTmp.nodes[node]['start']
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GaccLastTuneTmp.nodes[node]['tinx'] = 0
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['tinx'] = 0
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                        # case: red (red) green, the two red are not continuous
                    elif self.getModalCheck(pred, node) and not self.getModalCheck(node, success) and \
                         not self.getDependCheck(pred, node):
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=True)
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        # recursion, the next node
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                    # case: blue (red) green
                    elif not self.getModalCheck(node, success) and not self.getModalCheck(pred, node):
                        # dont need to change anything for this node, the weight trans is already tuned
                        # recursion, the next node
                        accObj = self.getAccObjbyName(self.GaccLastTuneTmp.nodes[node]['acc'], self.AccListLastTuneTmp)
                        layerInfo = self.GaccLastTuneTmp.nodes[node]['attri']
                        isBind = self.GaccLastTuneTmp.nodes[node]['bind']
                        if isBind:
                            WEItrans = False
                        else:
                            WEItrans = True
                        accObj.getRunPara(layerInfo)
                        LatSec, tin, tout = accObj.getLayerPerf(WEItrans=WEItrans, OFMtrans=True, IFMtrans=True)
                        self.GaccLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['start'] + LatSec
                        self.GtimeLastTuneTmp.nodes[node]['end'] = self.GaccLastTuneTmp.nodes[node]['end']
                        self.GaccLastTuneTmp.nodes[node]['tin'] = tin
                        self.GaccLastTuneTmp.nodes[node]['tout'] = tout
                        self.GtimeLastTuneTmp.nodes[node]['tin'] = tin
                        self.GtimeLastTuneTmp.nodes[node]['tout'] = tout
                        # recursion, the next node
                        self.getGraphUpdatViaModalLastTune(success)

                # self.Gtimeifm = copy.deepcopy(self.GtimeLastTuneTmp)


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
        fontsize = 7
        # **************************************************************************************************************
        colormap = {}
        for acc in self.AccList:
            accName = acc.getAccName()
            colormap[accName] = []
            colormap[accName].append(self.getColor())

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

        # **************************************************************************************************************
        # nodes = list(self.Gmap)
        # color_map = []
        # for node in nodes:
        #     if self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'CNN':
        #         color_map.append('red')
        #     elif self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'LSTM':
        #         color_map.append('blue')
        #     elif self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'FC':
        #         color_map.append('green')
        #     else:
        #         print('not supported computation model')
        #         print('bug in getMappingcom')
        #         return
        #
        # pos = nx.circular_layout(self.Gmap)
        # plt.subplot(321)
        # nx.draw(self.Gmap, pos, with_labels=True, edge_color='r', node_color=color_map, node_size=50, width=1, font_size=fontsize)

        plt.figure(2)
        plt.subplot(121 + i)
        pos = nx.circular_layout(self.Gmap, scale=1)

        nx.draw(self.Gmap, with_labels=True, edge_color='r',
                node_color=sum([colormap[self.Gtimeini.nodes[node]['acc']] for node in self.Gmap.nodes()], []),
                node_size=150, width=1, font_size=fontsize)

        for k, v in self.AccListini.items():
            if v:
                plt.plot([], [], label=k, color=''.join(colormap[k]))

        plt.title('mapping results', fontsize=10)
        figure = plt.gcf()
        figure.set_size_inches(20, 9)
        plt.legend(fontsize=fontsize, fancybox=True, framealpha=0.7)
        plt.savefig('0.svg', format='svg', dpi=100)
        # ************************************************************************
        f = plt.figure(0)

        # **************************************************************************************************************
        # nodes = list(self.Gmap)
        # color_map = []
        # for node in nodes:
        #     if self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'CNN':
        #         color_map.append('red')
        #     elif self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'LSTM':
        #         color_map.append('blue')
        #     elif self.Gmap.nodes[str(node)]['attri']['LayerType'] == 'FC':
        #         color_map.append('green')
        #     else:
        #         print('not supported computation model')
        #         print('bug in getMappingcom')
        #         return
        # pos = nx.circular_layout(self.Gmap)
        # plt.subplot(231)
        # nx.draw(self.Gmap, pos, with_labels=True, edge_color='r', node_color=color_map, node_size=50, width=1, font_size=fontsize)
        # plt.title('starter map', fontsize=10)
        # **************************************************************************************************************
        # nodes = list(self.Gmap)
        # colormapiteration = {}
        # colormap_iteration = []
        # for i in range(len(self.layers2Map.keys())):
        #     colormapiteration[i] = []
        #     colormapiteration[i].append(self.getColor())
        #
        # for node in nodes:
        #     for k, vs in self.layers2Map.items():
        #         for v in vs:
        #             if node == v:
        #                 colormap_iteration.append(colormapiteration[k])
        #
        # # print(self.layers2Map)
        # ax = plt.subplot(232)
        # colormap_iteration = sum(colormap_iteration, [])
        # pos = nx.circular_layout(self.Gmap)
        #
        # for label in colormapiteration:
        #     ax.plot(color=colormapiteration[label],label=label)
        #
        # # nx.draw(self.Gmapstart, pos, with_labels=True, edge_color='r', node_color=colormap_iteration, node_size=50, width=1,
        # #         font_size=fontsize)
        #
        # for k, vs in self.layers2Map.items():
        #     nx.draw(self.Gmap, pos, nodelist = vs, with_labels=True, edge_color='r', node_color=colormap_iteration[k], node_size=50,
        #             width=1, font_size=fontsize, label=k)
        #
        # plt.legend(fontsize=fontsize, fancybox=True, framealpha=0.7)
        # plt.title('mapping order for layers', fontsize=10)
        # **************************************************************************************************************
        # 此处是画加速器映射图


        # *****************************************************
        plt.subplot(121+i)
        pos = nx.circular_layout(self.Gmap, scale=1)

        # pos = nx.kamada_kawai_layout(self.Gtime, scale = 3)
        # pos = nx.spring_layout(self.Gtime, k=3/math.sqrt(self.Gmap.number_of_nodes()))

        nx.draw(self.Gmap, pos, with_labels=True, edge_color='r',
                node_color=sum([colormap[self.GtimeLastTune.nodes[node]['acc']]for node in self.Gmap.nodes()], []),
                node_size=150, width=1, font_size=fontsize)

        for k, v in self.AccLayersLastTune.items():
            if v:
                plt.plot([], [], label=k, color=''.join(colormap[k]))

        plt.title('mapping results', fontsize=10)
        figure = plt.gcf()
        figure.set_size_inches(20, 9)
        plt.legend(fontsize=fontsize, fancybox=True, framealpha=0.7)
        f.savefig('1.svg', format='svg', dpi = 100)
        # **************************************************************************************************************
        f = plt.figure(1)
        # **************************************************************************************************************
        # layertimemap = {}
        # nodes = self.Gacc.nodes
        # for node in nodes:
        #     layertimemap[node] = []
        #     # layertimemap[node].append(self.Gtimebind.nodes[node]['start'])
        #     # layertimemap[node].append(self.Gtimebind.nodes[node]['end'])
        #     # layertimemap[node].append(self.Gtimebind.nodes[node]['acc'])
        #     # layertimemap[node].append(self.Gtimeinibind.nodes[node]['start'])
        #     # layertimemap[node].append(self.Gtimeinibind.nodes[node]['end'])
        #     # layertimemap[node].append(self.Gtimeinibind.nodes[node]['acc'])
        #     layertimemap[node].append(self.Gtimeiniifm.nodes[node]['start'])
        #     layertimemap[node].append(self.Gtimeiniifm.nodes[node]['end'])
        #     layertimemap[node].append(self.Gtimeiniifm.nodes[node]['acc'])
        #     # layertimemap[node].append(self.Gtimeinimap.nodes[node]['start'])
        #     # layertimemap[node].append(self.Gtimeinimap.nodes[node]['end'])
        #     # layertimemap[node].append(self.Gtimeinimap.nodes[node]['acc'])
        # layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[0][1], x[1][0])))
        # plt.subplot(222+i*2)
        # for k, v in self.AccLayers.items():
        #     if v:
        #         plt.plot([], [], label=k, color=''.join(colormap[k]))
        # for k, v in layertimemap.items():
        #     plt.barh(y=k, width=v[1] - v[0], height=0.5, left=v[0], color=colormap[layertimemap[k][2]])
        # plt.yticks(fontsize=5)
        # plt.legend()
        # plt.title('detailed time for layers', fontsize=10)
        # **************************************************************************************************************
        layertimemap = {}
        nodes = self.Gmap.nodes
        for node in nodes:
            layertimemap[node] = []
            # layertimemap[node].append(self.Gtimeiniifm.nodes[node]['start'])
            # layertimemap[node].append(self.Gtimeiniifm.nodes[node]['end'])
            # layertimemap[node].append(self.Gtimeiniifm.nodes[node]['acc'])
            layertimemap[node].append(self.GtimeLastTune.nodes[node]['start'])
            layertimemap[node].append(self.GtimeLastTune.nodes[node]['end'])
            layertimemap[node].append(self.GtimeLastTune.nodes[node]['acc'])

        for k, v in self.AccLayers.items():
            if v:
                plt.plot([], [], label=k, color=''.join(colormap[k]))
        layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[0][1], x[1][0])))

        plt.subplot(221+i*2)
        for k, v in layertimemap.items():
            plt.barh(y=k, width=v[1]-v[0], height=0.5, left=v[0], color = colormap[layertimemap[k][2]])
        plt.yticks(fontsize=5)
        plt.legend()
        plt.title('detailed time for layers', fontsize=10)
        # **************************************************************************************************************
        layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[1][2], x[1][0])))
        plt.subplot(222+i*2)
        for k, v in layertimemap.items():
            # layerOrder = k.split("L")
            # print(layerOrder)
            plt.barh(y=v[2], width=v[1]-v[0], height=0.3, left=v[0], color = colormap[layertimemap[k][2]])
        plt.yticks(fontsize=8)  # y轴
        plt.title('operation time for accelerators', fontsize=10)

# **************************************************************************************************************
        # layertimemap = {}
        # nodes = self.Gacc.nodes
        # for node in nodes:
        #     layertimemap[node] = []
        #     layertimemap[node].append(self.Gtimeaftermapping.nodes[node]['start'])
        #     layertimemap[node].append(self.Gtimeaftermapping.nodes[node]['end'])
        #     layertimemap[node].append(self.Gtimeaftermapping.nodes[node]['acc'])
        #
        # layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[0][1], x[1][0])))
        #
        # plt.subplot(325)
        # for k, v in layertimemap.items():
        #     plt.barh(y=k, width=v[1]-v[0], height=0.5, left=v[0], color = colormap[layertimemap[k][2]])
        # plt.yticks(fontsize=5)
        # plt.legend(fontsize=5)

        # **************************************************************************************************************
        # layertimemap = dict(sorted(layertimemap.items(), key=lambda x: (x[1][2], x[1][0])))
        # plt.subplot(326)
        # for k, v in layertimemap.items():
        #     # layerOrder = k.split("L")
        #     # print(layerOrder)
        #     plt.barh(y=v[2], width=v[1]-v[0], height=0.3, left=v[0], color = colormap[layertimemap[k][2]])
        # plt.yticks(fontsize=8)  # y轴
        # plt.figure(2)
        # pos = nx.circular_layout(self.Amap)
        # nx.draw(self.Amap, pos, with_labels=True, edge_color='r', node_color='r', node_size=50,
        #         width=[float(v['weight'] / 10) for (r, c, v) in self.Amap.edges(data=True)], font_size=5)
        # labels = nx.get_edge_attributes(self.Amap, 'weight')
        # nx.draw_networkx_edge_labels(self.Amap, pos, edge_labels=labels)
        figure = plt.gcf()
        figure.set_size_inches(16, 10)
        f.savefig('2.svg',format='svg', dpi =100)


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
    AccsList4 = [acc1, acc2, fc_acc_ese()]
    AccsList5 = [acc1, acc2, acc3, acc4, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13]
    # AccsLists = [AccsList4, AccsList3]
    AccsLists = [AccsList5]
    Modalities = [Modality7]

    optlat = {}
    optenu = {}
    h2hlat = {}
    h2henu = {}
    h2plat = {}
    h2penu = {}

    i = 0
    for Modality in Modalities:
        for AccsList in AccsLists:
            optlat[i] = [0]
            optenu[i] = [0]
            h2plat[i] = [0]
            h2penu[i] = [0]
            print(AccsList)
            print('new accelerator combs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            NetBand = 0.125
            DDR_WEI = 0.5
            Dtype = 32
            fontsize = 6
            for Acc in AccsList:
                Acc.NetBand = NetBand
                Acc.DDR_WEI = DDR_WEI
            print('Modality case 1')
            # InitMapper = MapperInit()
            # # InitMapper.Gmap.add_edge('M2L5', 'M1L6')
            # InitMapper.NetBand = NetBand
            # InitMapper.Dtype = Dtype
            # InitMapper.getModalityLayers(Modality)
            # InitMapper.getAccList(AccsList)

            # start = time.time()
            # InitMapper.getLayerOrder()
            # InitMapper.getAccListPermu()
            # InitMapper.getMappingOptimal(start, optlat[i], optenu[i])
            # end = time.time()
            # print('the enumerating time is:', end - start)

            InitMapper = MapperInit()
            InitMapper.NetBand = NetBand
            InitMapper.Dtype = Dtype
            # *******************************************
            InitMapper.getModalityLayers(Modality)
            InitMapper.getAccList(AccsList)
            start = time.time()
            inistart = time.time()
            InitMapper.getMappingcom(Modality)
            iniend = time.time()
            InitMapper.getKnapsack()
            InitMapper.getBindedTime()
            InitMapper.getIfmOfmTrans()
            InitMapper.getHomoNeighbor()
            end = time.time()
            print('the ini time is:', iniend - inistart, '\n')
            print('the H2H time is:', end - start, '\n')
            InitMapper.getperftable(0)
            # plt.show()

            InitMapper.getModalityLayers(Modality)
            InitMapper.getAccList(AccsList)
            InitMapper.getLayerOrder()
            start = time.time()
            InitMapper.getMapping()
            InitMapper.getKnapsack()
            InitMapper.getBindedTime()
            InitMapper.getIfmOfmTrans()
            InitMapper.getHomoNeighborh2h()

            end = time.time()
            print('the H2H time is:', end - start)
            InitMapper.getModalityLayers(Modality)
            InitMapper.getperftable(1)

            # plt.show()
