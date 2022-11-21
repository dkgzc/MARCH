from math import *

class cnn_acc_7:
    # GX1150
    # bandwidth was 512
    # NetBand=5GB/s
    # DDR = 2GB
    # DDR_WEI = 80%
    # Dtype = 32bit
    # Freq = 150
    # Bandwidth = 512
    def __init__(self, throughput=130, NetBand=0.5, DDR=4, DDR_WEI=0.8, Dtype=32, Freq=150, Bandwidth=512, Tm=32, Tn=32,
                 Tr=16, Tc=14, Tr_=16, Tc_=16, power=21.2):
        # the parameters that need to be transferred in
        self.M = None
        self.N = None
        self.R = None
        self.C = None
        self.K = None
        self.S = None  # Stride
        self.P = None  # Stride
        # the parameters that is fixed for an accelerator
        self.NetBand = NetBand
        self.DDR = DDR
        self.DDR_WEI = DDR_WEI
        self.Dtype = Dtype
        self.Freq = Freq
        self.Bandwidth = Bandwidth
        self.Tm = Tm
        self.Tn = Tn
        self.Tr = Tr
        self.Tc = Tc
        self.Tr_ = Tr_
        self.Tc_ = Tc_
        self.isPingPong = True
        self.accName = 'cnn_acc_7'
        self.accType = 'CNN'
        # represent the number of extra weight element
        self.power = power
        self.extraWeightElement = self.DDR * self.DDR_WEI * 8 * (10 ** 9) / self.Dtype
        self.assiLayers = []
        self.bindLayer = []
        # Gop/s
        self.Th = throughput

    def getExtraWeightElement(self):
        self.extraWeightElement = self.DDR * self.DDR_WEI * 8 * (10 ** 9) / self.Dtype

    def getPower(self):
        return self.power

    def getRunPara(self, Para):
        self.M = Para['M']
        self.N = Para['N']
        self.R = Para['R']
        self.C = Para['C']
        self.K = Para['K']
        self.S = Para['S']  # Stride
        self.P = Para['P']  # Padding
        self.LayerName = Para['LayerName']

    def getAccName(self):
        return self.accName

    def getOps(self):
        return 2 * self.R * self.C * self.M * self.K * self.K * self.N

    def getTileOps(self):
        return 2 * self.Tr * self.Tc * self.Tm * self.K * self.K * self.Tn

    def getCompPerf(self):
        ops = self.getTileOps() / (10 ** 9)
        tComp = ops / self.Th
        # the resturned tComp is in second
        return tComp

    def getDDR2AccPerfPerTile(self):
        DDR2AccBand = self.Bandwidth * self.Freq * (10 ** 6)
        tWeight = (self.Tm * self.Tn * self.K * self.K) \
                  * self.Dtype / DDR2AccBand
        tIfm = (self.Tn * self.Tr_ * self.Tc_) \
               * self.Dtype / DDR2AccBand
        tDataIn = tWeight + tIfm
        tDataOut = (self.Tm * self.Tr * self.Tc) * self.Dtype / DDR2AccBand
        return tDataIn, tDataOut

    def getDataInPerf(self, WEItrans, IFMtrans, OFMtrans):
        # convert GB/s to bit/second
        DDR2MemBand = self.NetBand * (10 ** 9) * 8
        if WEItrans == True and IFMtrans == True:
            tWeight = (self.M * self.N * self.K * self.K) \
                      * self.Dtype / DDR2MemBand
            tIfm = (self.N * self.R * 2 * self.C * 2) \
                   * self.Dtype / DDR2MemBand
            tDataIn = tWeight + tIfm
        if WEItrans == False and IFMtrans == True:
            tDataIn = (self.N * self.R * 2 * self.C * 2) * self.Dtype / DDR2MemBand
        if WEItrans == True and IFMtrans == False:
            tDataIn = (self.M * self.N * self.K * self.K) \
                      * self.Dtype / DDR2MemBand
        if WEItrans == False and IFMtrans == False:
            tDataIn = 0

        if OFMtrans == True:
            tDataOut = (self.M * self.R * self.C) \
                       * self.Dtype / DDR2MemBand
        if OFMtrans == False:
            tDataOut = 0
        return tDataIn, tDataOut

    def getLayerPerf(self, WEItrans=True, IFMtrans=True, OFMtrans=True):

        # layer transTime
        tDataInfromMem, tDataOutfromMem = self.getDataInPerf(WEItrans, IFMtrans, OFMtrans)
        # process a layer tile by tile
        tDataIn, tDataOut = self.getDDR2AccPerfPerTile()
        tComp = self.getCompPerf()
        # we use output stationary, send in some ifm, we compute the partial result of a ofm tile, using ping-pong buffer
        Lat1 = tDataIn + tComp
        # after accumenlating all partial results of a ofm tile, this ofm tile computations are done
        # Lat2 = (ceil(self.N / self.Tn) - 1) * Lat1 + max((tDataIn + tDataOut), tComp)
        Lat2 = ceil(self.N / self.Tn) * Lat1 + tDataOut
        # after computing all ofm tiles, a conv layer is done
        LatSec = ceil(self.R / self.Tr) * ceil(self.C / self.Tc) * ceil(self.M / self.Tm) * Lat2

        LatSec = LatSec + tDataInfromMem + tDataOutfromMem
        return LatSec, tDataInfromMem, tDataOutfromMem

    def getDataTranPerf(self, bandwidth, WEItrans, IFMtrans, OFMtrans):
        # convert GB/s to bit/second
        DDR2MemBand = self.NetBand * (10**9) * 8
        DDR2MemBand = bandwidth * (10 ** 9) * 8
        if WEItrans==True and IFMtrans==True:
            tWeight = (self.M * self.N * self.K * self.K) \
                  * self.Dtype / DDR2MemBand
            tIfm = (self.N * self.R * 2 * self.C * 2) \
                  * self.Dtype / DDR2MemBand
            tDataIn = tWeight + tIfm
        if WEItrans==False and IFMtrans==True:
            tDataIn = ( self.N * self.R * 2 * self.C * 2) * self.Dtype / DDR2MemBand
        if WEItrans==True and IFMtrans==False:
            tDataIn = (self.M * self.N * self.K * self.K) \
                  * self.Dtype / DDR2MemBand
        if WEItrans==False and IFMtrans==False:
            tDataIn = 0

        if OFMtrans==True:
            tDataOut = (self.M * self.R * self.C ) \
                  * self.Dtype / DDR2MemBand
        if OFMtrans == False:
            tDataOut = 0
        return tDataIn, tDataOut

    def getLayerPerf1(self, WEItrans=False,IFMtrans=False, OFMtrans=False):

        # layer transTime
        # process a layer tile by tile
        # tDataInfromMem, tDataOutfromMem = self.getDataTranPerf(bandwidth, WEItrans, IFMtrans, OFMtrans)
        tDataIn, tDataOut = self.getDDR2AccPerfPerTile()
        tComp = self.getCompPerf()
        #we use output stationary, send in some ifm, we compute the partial result of a ofm tile, using ping-pong buffer
        Lat1 = tDataIn + tComp
        #after accumenlating all partial results of a ofm tile, this ofm tile computations are done
        # Lat2 = (ceil(self.N / self.Tn) - 1) * Lat1 + max((tDataIn + tDataOut), tComp)
        Lat2 = ceil(self.N / self.Tn) * Lat1 + tDataOut
        #after computing all ofm tiles, a conv layer is done
        LatSec = ceil(self.R / self.Tr) * ceil(self.C / self.Tc) * ceil(self.M / self.Tm) * Lat2

        # LatSec = LatSec + tDataInfromMem + tDataOutfromMem

        return LatSec, self.Th

    def getp2p(self, accslist):
        p2p = {}
        for acc in accslist:
            accname = acc.getAccName()
            p2p[accname] = 0.125
        return p2p

if __name__ == "__main__":
    acc=cnn_acc_7()
    M2L2 = {'LayerType': 'CNN', 'LayerName': 'M2L2', 'M': 128, 'N': 192, 'R': 13, 'C': 13, 'K': 3, 'S': 1, 'P': 1}
    acc.getRunPara(M2L2)
    LatSec, Th= acc.getLayerPerf()
    print(LatSec, Th, '\n')






