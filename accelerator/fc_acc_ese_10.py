from math import *

class fc_acc_ese:
    # XCKU060
    # bandwidth was 512
    # NetBand=5GB/s
    # DDR = 2GB
    # DDR_WEI = 80%
    # Dtype = 32bit
    # Freq = 250
    # Bandwidth = 512
    # throughput was 282, scale to 28.2
    def __init__(self,  throughput=282, NetBand=0.5, DDR=1, DDR_WEI=0.8, Dtype=32, Freq=150, Bandwidth=512,  power = 41):
        # [MxN] [NxK] [MxN] is weight
        self.M = None
        self.N = None
        self.K = None
        self.NetBand = NetBand
        self.DDR = DDR
        self.DDR_WEI = DDR_WEI
        self.Dtype = Dtype
        self.Freq = Freq
        self.Bandwidth = Bandwidth
        # S_k is a unique acc parameter in this paper, it represents the parallel size
        self.isPingPong = True
        self.accName = 'FCAccESE'
        self.accType = 'FC'
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
        self.M = Para['M']
        self.N = Para['N']
        self.K = Para['K']
        self.LayerName = Para['LayerName']

    def getAccName(self):
        return self.accName

    def getOps(self):
        return self.N *self.M * self.K * 2

    def getCompPerf(self):
        ops = self.getOps()
        tComp = ops / (10**9) / self.Th
        return tComp

    def getDDR2AccPerfPerTile(self):
        DDR2AccBand = self.Bandwidth * self.Freq * (10 ** 6)
        tWeight = (self.M*self.N) \
                  * self.Dtype / DDR2AccBand
        tIfm = (self.N * self.K) * self.Dtype / DDR2AccBand
        tDataIn = tWeight + tIfm
        tDataOut = (self.M * self.K) * self.Dtype / DDR2AccBand
        return tDataIn, tDataOut

    def getDataInPerf(self, WEItrans, IFMtrans, OFMtrans):
        # convert GB/s to bit/second
        DDR2MemBand = self.NetBand * (10 ** 9) * 8
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

    def getLayerPerf(self, WEItrans=True,IFMtrans=True, OFMtrans=True):
        # operation of a layer
        # layer transTime
        tDataInfromMem, tDataOutfromMem = self.getDataInPerf(WEItrans, IFMtrans, OFMtrans)
        # process a layer tile by tile
        tDataIn, tDataOut = self.getDDR2AccPerfPerTile()
        tComp = self.getCompPerf()
        LatSec = max(tDataIn, tComp, tDataOut) + tDataInfromMem + tDataOutfromMem
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
    acc=fc_acc_ese()

    Para={'M':512, 'N':512, 'K':1, 'LayerName':'fc1'}
    acc.getRunPara(Para)
    LatSec, Th= acc.getLayerPerf()
    print(LatSec, Th, '\n')




