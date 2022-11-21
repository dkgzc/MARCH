#cite paper 'Achieving Full Parallelism in LSTM via a UnifiedAccelerator Design'

from math import *

class lstm_acc_vc707:
    # vc707
    # bandwidth was 512
    # NetBand=5GB/s
    # DDR = 0.256
    # DDR_WEI = 80%
    # Dtype = 32bit
    # Freq = 200
    # Bandwidth = 256
    # bandwidth was 256
    def __init__(self, throughput=10, NetBand=0.5, DDR=1, DDR_WEI=0.8, Dtype=32, Freq=150, Bandwidth=512,
                 power=19.63):
        self.Hidden = None
        self.Dimen = None
        self.Iter = None
        self.NetBand = NetBand
        self.DDR = DDR
        self.DDR_WEI = DDR_WEI
        self.Dtype = Dtype
        self.Freq = Freq
        self.Bandwidth = Bandwidth
        # S_k is a unique acc parameter in this paper, it represents the parallel size
        self.isPingPong = True
        self.accName = 'LSTMAccVC707'
        self.accType = 'LSTM'
        self.power = power
        self.extraWeightElement = self.DDR * self.DDR_WEI * 8 * (10 ** 9) / self.Dtype
        self.assiLayers = []
        self.bindLayer = []
        self.Th = throughput

    def getExtraWeightElement(self):
        self.extraWeightElement = self.DDR * self.DDR_WEI * 8 * (10 ** 9) / self.Dtype

    def getPower(self):
        # power should be determined by the parallel size, etc
        # use a dummy data here
        return self.power

    def getRunPara(self, Para):
        self.Hidden = Para['Hidden']
        self.Dimen = Para['Dimen']
        self.Iter = Para['Iter']
        self.LayerName = Para['LayerName']

    def getAccName(self):
        return self.accName

    def getOps(self):
        return 4 * self.Hidden * self.Dimen * 2 + 4 * self.Hidden * self.Hidden * 2

    def getCompPerf(self):
        ops = self.getOps()
        tComp = ops / (10 ** 9) / self.Th
        return tComp

    def getDDR2AccPerfPerTile(self):
        DDR2AccBand = self.Bandwidth * self.Freq * (10 ** 6)
        tWeight = (4 * self.Hidden * self.Dimen + 4 * self.Hidden * self.Hidden) * self.Dtype / DDR2AccBand
        tIfm = (self.Dimen) * self.Dtype / DDR2AccBand
        tDataIn = tWeight + tIfm
        tDataOut = (self.Hidden) * self.Dtype / DDR2AccBand
        return tDataIn, tDataOut

    def getDataInPerf(self, WEItrans, IFMtrans, OFMtrans):
        DDR2MemBand = self.NetBand * (10 ** 9) * 8
        if WEItrans == True and IFMtrans == True:
            tDataIn = (4 * self.Hidden * self.Dimen + 4 * self.Hidden * self.Hidden + self.Dimen) \
                      * self.Dtype / DDR2MemBand
        elif WEItrans == False and IFMtrans == True:
            tDataIn = (self.Dimen) * self.Dtype / DDR2MemBand
        elif WEItrans == True and IFMtrans == False:
            tDataIn = (4 * self.Hidden * self.Dimen + 4 * self.Hidden * self.Hidden) * self.Dtype / BandwidthFreq
        elif WEItrans == False and IFMtrans == False:
            tDataIn = 0

        if OFMtrans == True:
            tDataOut = (self.Hidden) * self.Dtype / DDR2MemBand
        if OFMtrans == False:
            tDataOut = 0
        return tDataIn, tDataOut

    def getLayerPerf(self, WEItrans=True, IFMtrans=True, OFMtrans=True):
        # operation of a layer
        # process a layer tile by tile
        # layer transTime
        tDataInfromMem, tDataOutfromMem = self.getDataInPerf(WEItrans, IFMtrans, OFMtrans)
        # process a layer tile by tile
        tDataIn, tDataOut = self.getDDR2AccPerfPerTile()
        tComp = self.getCompPerf()

        LatSec = self.Iter * max(tDataIn, tComp, tDataOut) + tDataInfromMem + tDataOutfromMem
        return LatSec, tDataInfromMem, tDataOutfromMem

    def getDataTranPerf(self, bandwidth, WEItrans, IFMtrans, OFMtrans):
        # convert GB/s to bit/second
        DDR2MemBand = bandwidth * (10 ** 9) * 8
        if WEItrans==True and IFMtrans==True:
            tDataIn = (self.M*self.N + self.N*self.K) \
                  * self.Dtype / DDR2MemBand
        elif WEItrans==False and IFMtrans==True:
            tDataIn = (self.N*self.K)* self.Dtype / DDR2MemBand
        elif WEItrans==True and IFMtrans==False:
            tDataIn = (self.M*self.N) * self.Dtype / DDR2MemBand
        elif WEItrans==False and IFMtrans==False:
            tDataIn = 0

        if OFMtrans == True:
            tDataOut =  (self.M * self.K) * self.Dtype / DDR2MemBand
        if OFMtrans == False:
            tDataOut = 0
        return tDataIn, tDataOut

    def getLayerPerf1(self, WEItrans=False,IFMtrans=False, OFMtrans=False):

        # layer transTime
        # process a layer tile by tile
        # tDataInfromMem, tDataOutfromMem = self.getDataTranPerf(bandwidth, WEItrans, IFMtrans, OFMtrans)
        tDataIn, tDataOut = self.getDDR2AccPerfPerTile()
        tComp = self.getCompPerf()
        # LatSec = max(tDataIn, tComp, tDataOut) + tDataInfromMem + tDataOutfromMem
        LatSec = max(tDataIn, tComp, tDataOut)
        return LatSec, self.Th

    def getp2p(self, accslist):
        p2p = {}
        for acc in accslist:
            accname = acc.getAccName()
            p2p[accname] = 0.125
        return p2p
if __name__ == "__main__":
    acc=lstm_acc_vc707()

    Para={'Hidden':512, 'Dimen':512, 'Iter':1, 'LayerName':'lstm1'}
    acc.getRunPara(Para)
    LatSec, Th= acc.getLayerPerf()
    print(LatSec, Th, '\n')



