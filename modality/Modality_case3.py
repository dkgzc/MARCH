# modality 1
# vgg16
M1L1 = {'LayerType': 'CNN', 'LayerName': 'M1L1', 'M': 64,  'N': 3, 'R': 224, 'C': 224, 'K': 3, 'S': 1, 'P': 1}
M1L2 = {'LayerType': 'CNN', 'LayerName': 'M1L2', 'M': 64, 'N': 64, 'R': 224, 'C': 224, 'K': 3, 'S': 1, 'P': 1}

M1L3 = {'LayerType': 'CNN', 'LayerName': 'M1L3', 'M': 128,  'N': 64, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}
M1L4 = {'LayerType': 'CNN', 'LayerName': 'M1L4', 'M': 128, 'N': 128, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}

M1L5 = {'LayerType': 'CNN', 'LayerName': 'M1L5', 'M': 256,  'N': 128, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M1L6 = {'LayerType': 'CNN', 'LayerName': 'M1L6', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M1L7 = {'LayerType': 'CNN', 'LayerName': 'M1L7', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}


M1L8 = {'LayerType': 'CNN', 'LayerName': 'M1L8', 'M': 512, 'N': 256, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M1L9 = {'LayerType': 'CNN', 'LayerName': 'M1L9', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M1L10 = {'LayerType': 'CNN', 'LayerName': 'M1L10', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}

M1L11 = {'LayerType': 'CNN', 'LayerName': 'M1L11', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}
M1L12 = {'LayerType': 'CNN', 'LayerName': 'M1L12', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}
M1L13 = {'LayerType': 'CNN', 'LayerName': 'M1L13', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}

M1L14 = {'LayerType': 'FC',  'LayerName': 'M1L14', 'M':4096, 'N':25088, 'K':1 }
M1L15 = {'LayerType': 'FC',  'LayerName': 'M1L15', 'M':4096, 'N':4096, 'K':1 }
M1L16 = {'LayerType': 'FC',  'LayerName': 'M1L16', 'M':1000, 'N':4096, 'K':1 }

M1 = {'M1L1': M1L1, 'M1L2': M1L2, 'M1L3': M1L3
    ,
      'M1L4': M1L4, 'M1L5': M1L5,
      'M1L6': M1L6, 'M1L7': M1L7, 'M1L8': M1L8,       'M1L9': M1L9, 'M1L10': M1L10,
      'M1L11': M1L11, 'M1L12': M1L12, 'M1L13': M1L13, 'M1L14': M1L14, 'M1L15': M1L15,
      'M1L16': M1L16
      }

#VDCNN: I convert the comp. to conv2D
# e.g 3*64 conv kernel convert to3*3 kernel
# convert to a conv with IFM_channel =64 (but channels are duplicated) weight kernel =64, OFM_channel=1
M2L1 = {'LayerType': 'CNN', 'LayerName': 'M2L1', 'M': 1,  'N': 64, 'R': 1014, 'C': 21, 'K': 3, 'S': 1, 'P': 1}
M2L2 = {'LayerType': 'CNN', 'LayerName': 'M2L2', 'M': 1, 'N': 64, 'R': 1014, 'C': 21, 'K': 3, 'S': 1, 'P': 1}

M2L3 = {'LayerType': 'CNN', 'LayerName': 'M2L3', 'M': 1, 'N': 128, 'R': 507, 'C': 42, 'K': 3, 'S': 1, 'P': 1}
M2L4 = {'LayerType': 'CNN', 'LayerName': 'M2L4', 'M': 1, 'N': 128, 'R': 507, 'C': 42, 'K': 3, 'S': 1, 'P': 1}

M2L5 = {'LayerType': 'CNN', 'LayerName': 'M2L5', 'M': 1, 'N': 254, 'R': 254, 'C': 85, 'K': 3, 'S': 1, 'P': 1}
M2L6 = {'LayerType': 'CNN', 'LayerName': 'M2L6', 'M': 1, 'N': 254, 'R': 254, 'C': 85, 'K': 3, 'S': 1, 'P': 1}

M2L7 = {'LayerType': 'CNN', 'LayerName': 'M2L7', 'M': 1, 'N': 256, 'R': 127, 'C': 171, 'K': 3, 'S': 1, 'P': 1}
M2L8 = {'LayerType': 'CNN', 'LayerName': 'M2L8', 'M': 1, 'N': 256, 'R': 127, 'C': 171, 'K': 3, 'S': 1, 'P': 1}

M2L9 = {'LayerType': 'FC',  'LayerName': 'M1L14', 'M':4096, 'N':21717, 'K':1 }

M2 = {'M2L1': M2L1, 'M2L2': M2L2, 'M2L3': M2L3
    ,      'M2L4': M2L4
    , 'M2L5': M2L5,
      'M2L6': M2L6, 'M2L7': M2L7, 'M2L8': M2L8,       'M2L9': M2L9
      }

# modality 3
# vgg16
M3L1 = {'LayerType': 'CNN', 'LayerName': 'M3L1', 'M': 64,  'N': 3, 'R': 224, 'C': 224, 'K': 3, 'S': 1, 'P': 1}
M3L2 = {'LayerType': 'CNN', 'LayerName': 'M3L2', 'M': 64, 'N': 64, 'R': 224, 'C': 224, 'K': 3, 'S': 1, 'P': 1}

M3L3 = {'LayerType': 'CNN', 'LayerName': 'M3L3', 'M': 128,  'N': 64, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}
M3L4 = {'LayerType': 'CNN', 'LayerName': 'M3L4', 'M': 128, 'N': 128, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}

M3L5 = {'LayerType': 'CNN', 'LayerName': 'M3L5', 'M': 256,  'N': 128, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M3L6 = {'LayerType': 'CNN', 'LayerName': 'M3L6', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M3L7 = {'LayerType': 'CNN', 'LayerName': 'M3L7', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}


M3L8 = {'LayerType': 'CNN', 'LayerName': 'M3L8', 'M': 512, 'N': 256, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M3L9 = {'LayerType': 'CNN', 'LayerName': 'M3L9', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M3L10 = {'LayerType': 'CNN', 'LayerName': 'M3L10', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}

M3L11 = {'LayerType': 'CNN', 'LayerName': 'M3L11', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}
M3L12 = {'LayerType': 'CNN', 'LayerName': 'M3L12', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}
M3L13 = {'LayerType': 'CNN', 'LayerName': 'M3L13', 'M': 512, 'N': 512, 'R': 14, 'C': 14, 'K': 3, 'S': 1, 'P': 1}

M3L14 = {'LayerType': 'FC',  'LayerName': 'M3L14', 'M':4096, 'N':25088, 'K':1 }
M3L15 = {'LayerType': 'FC',  'LayerName': 'M3L15', 'M':4096, 'N':4096, 'K':1 }
M3L16 = {'LayerType': 'FC',  'LayerName': 'M3L16', 'M':1000, 'N':4096, 'K':1 }

M3 = {'M3L1': M3L1, 'M3L2': M3L2
    , 'M3L3': M3L3
    ,
      'M3L4': M3L4, 'M3L5': M3L5,
      'M3L6': M3L6, 'M3L7': M3L7, 'M3L8': M3L8,       'M3L9': M3L9, 'M3L10': M3L10,
      'M3L11': M3L11, 'M3L12': M3L12, 'M3L13': M3L13, 'M3L14': M3L14,
        'M3L15': M3L15, 'M3L16': M3L16
      }

#VDCNN: I convert the comp. to conv2D
# e.g 3*64 conv kernel convert to3*3 kernel
# convert to a conv with IFM_channel =64 (but channels are duplicated) weight kernel =64, OFM_channel=1
M4L1 = {'LayerType': 'CNN', 'LayerName': 'M4L1', 'M': 1,  'N': 64, 'R': 1014, 'C': 21, 'K': 3, 'S': 1, 'P': 1}
M4L2 = {'LayerType': 'CNN', 'LayerName': 'M4L2', 'M': 1, 'N': 64, 'R': 1014, 'C': 21, 'K': 3, 'S': 1, 'P': 1}

M4L3 = {'LayerType': 'CNN', 'LayerName': 'M4L3', 'M': 1, 'N': 128, 'R': 507, 'C': 42, 'K': 3, 'S': 1, 'P': 1}
M4L4 = {'LayerType': 'CNN', 'LayerName': 'M4L4', 'M': 1, 'N': 128, 'R': 507, 'C': 42, 'K': 3, 'S': 1, 'P': 1}

M4L5 = {'LayerType': 'CNN', 'LayerName': 'M4L5', 'M': 1, 'N': 254, 'R': 254, 'C': 85, 'K': 3, 'S': 1, 'P': 1}
M4L6 = {'LayerType': 'CNN', 'LayerName': 'M4L6', 'M': 1, 'N': 254, 'R': 254, 'C': 85, 'K': 3, 'S': 1, 'P': 1}

M4L7 = {'LayerType': 'CNN', 'LayerName': 'M4L7', 'M': 1, 'N': 256, 'R': 127, 'C': 171, 'K': 3, 'S': 1, 'P': 1}
M4L8 = {'LayerType': 'CNN', 'LayerName': 'M4L8', 'M': 1, 'N': 256, 'R': 127, 'C': 171, 'K': 3, 'S': 1, 'P': 1}

M4 = {'M4L1': M4L1, 'M4L2': M4L2
    , 'M4L3': M4L3,      'M4L4': M4L4, 'M4L5': M4L5,
      'M4L6': M4L6, 'M4L7': M4L7, 'M4L8': M4L8
      }

Modality3 = [M1, M2, M3, M4]
# InitMapper.Gmap.add_edge('M4L8', 'M3L14')
if __name__ == "__main__":
    import ParaCount
    paraNum = ParaCount.paraCount(Modality3)
    print(paraNum/1000000)