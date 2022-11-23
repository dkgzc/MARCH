# input dimen (100, 34) 100 is length, 34 is embedded length, bidirectional make ite = 100*2
M1L1 = {'LayerType': 'LSTM','LayerName': 'M1L1', 'Hidden': 128, 'Dimen': 34, 'Iter': 200}
M1L2 = {'LayerType': 'LSTM','LayerName': 'M1L2', 'Hidden': 256, 'Dimen': 34, 'Iter': 200}
M1L3 = {'LayerType': 'FC',  'LayerName': 'M1L3', 'M':256, 'N':1024, 'K':1 }

M1 = {'M1L1': M1L1, 'M1L2': M1L2, 'M1L3': M1L3
      }

M2L1 = {'LayerType': 'LSTM','LayerName': 'M2L1', 'Hidden': 256, 'Dimen': 300, 'Iter': 1000}
M2L2 = {'LayerType': 'LSTM','LayerName': 'M2L2', 'Hidden': 512, 'Dimen': 300, 'Iter': 1000}
M2L3 = {'LayerType': 'FC',  'LayerName': 'M2L3', 'M':256, 'N':2048, 'K':1 }

M2 = {'M2L1': M2L1, 'M2L2': M2L2, 'M2L3': M2L3
      }

M3L1 = {'LayerType': 'CNN', 'LayerName': 'M3L1', 'M': 32,  'N': 1, 'R': 100, 'C': 99, 'K': 3, 'S': 1, 'P': 1}
M3L2 = {'LayerType': 'CNN', 'LayerName': 'M3L2', 'M': 64,  'N': 32, 'R': 50, 'C': 44, 'K': 3, 'S': 1, 'P': 1}
M3L3 = {'LayerType': 'CNN', 'LayerName': 'M3L3', 'M': 64,  'N': 64, 'R': 25, 'C': 22, 'K': 3, 'S': 1, 'P': 1}
M3L4 = {'LayerType': 'CNN', 'LayerName': 'M3L4', 'M': 128,  'N': 64, 'R': 12, 'C': 11, 'K': 3, 'S': 1, 'P': 1}
M3L5 = {'LayerType': 'CNN', 'LayerName': 'M3L5', 'M': 128,  'N': 128, 'R': 6, 'C': 6, 'K': 3, 'S': 1, 'P': 1}
M3L6 = {'LayerType': 'FC',  'LayerName': 'M3L6', 'M':256, 'N':768, 'K':1 }
M3L7 = {'LayerType': 'FC',  'LayerName': 'M3L7', 'M':256, 'N':768, 'K':1 }

M3 = {'M3L1': M3L1, 'M3L2': M3L2, 'M3L3': M3L3,      'M3L4': M3L4
    , 'M3L5': M3L5
    ,
      'M3L6': M3L6, 'M3L7': M3L7
      }

M4L1 = {'LayerType': 'CNN', 'LayerName': 'M4L1', 'M': 64,  'N': 3, 'R': 224, 'C': 224, 'K': 1, 'S': 1, 'P': 1}
M4L2 = {'LayerType': 'CNN', 'LayerName': 'M4L2', 'M': 64, 'N': 64, 'R': 224, 'C': 224, 'K': 3, 'S': 1, 'P': 1}
M4L3 = {'LayerType': 'CNN', 'LayerName': 'M4L3', 'M': 256, 'N': 64, 'R': 224, 'C': 224, 'K': 1, 'S': 1, 'P': 1}
# Resx3
M4L4 = {'LayerType': 'CNN', 'LayerName': 'M4L4', 'M': 128, 'N': 256, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
M4L5 = {'LayerType': 'CNN', 'LayerName': 'M4L5', 'M': 128, 'N': 128, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}
M4L6 = {'LayerType': 'CNN', 'LayerName': 'M4L6', 'M': 512, 'N': 128, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
M4L7 = {'LayerType': 'CNN', 'LayerName': 'M4L7', 'M': 128, 'N': 512, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
M4L8 = {'LayerType': 'CNN', 'LayerName': 'M4L8', 'M': 128, 'N': 128, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}
M4L9 = {'LayerType': 'CNN', 'LayerName': 'M4L9', 'M': 512, 'N': 128, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
M4L10 = {'LayerType': 'CNN', 'LayerName': 'M4L10', 'M': 128, 'N': 512, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
M4L11 = {'LayerType': 'CNN', 'LayerName': 'M4L11', 'M': 128, 'N': 128, 'R': 112, 'C': 112, 'K': 3, 'S': 1, 'P': 1}
M4L12 = {'LayerType': 'CNN', 'LayerName': 'M4L12', 'M': 512, 'N': 128, 'R': 112, 'C': 112, 'K': 1, 'S': 1, 'P': 1}
# Resx4
M4L13 = {'LayerType': 'CNN', 'LayerName': 'M4L13', 'M': 256, 'N': 512, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L14 = {'LayerType': 'CNN', 'LayerName': 'M4L14', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M4L15 = {'LayerType': 'CNN', 'LayerName': 'M4L15', 'M': 1024, 'N': 256, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L16 = {'LayerType': 'CNN', 'LayerName': 'M4L16', 'M': 256, 'N': 1024, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L17 = {'LayerType': 'CNN', 'LayerName': 'M4L17', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M4L18 = {'LayerType': 'CNN', 'LayerName': 'M4L18', 'M': 1024, 'N': 256, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L19 = {'LayerType': 'CNN', 'LayerName': 'M4L19', 'M': 256, 'N': 1024, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L20 = {'LayerType': 'CNN', 'LayerName': 'M4L20', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M4L21 = {'LayerType': 'CNN', 'LayerName': 'M4L21', 'M': 1024, 'N': 256, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L22 = {'LayerType': 'CNN', 'LayerName': 'M4L22', 'M': 256, 'N': 1024, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
M4L23 = {'LayerType': 'CNN', 'LayerName': 'M4L23', 'M': 256, 'N': 256, 'R': 56, 'C': 56, 'K': 3, 'S': 1, 'P': 1}
M4L24 = {'LayerType': 'CNN', 'LayerName': 'M4L24', 'M': 1024, 'N': 256, 'R': 56, 'C': 56, 'K': 1, 'S': 1, 'P': 1}
# Resx5
M4L25 = {'LayerType': 'CNN', 'LayerName': 'M4L25', 'M': 512, 'N': 1024, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L26 = {'LayerType': 'CNN', 'LayerName': 'M4L26', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M4L27 = {'LayerType': 'CNN', 'LayerName': 'M4L27', 'M': 2048, 'N': 512, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L28 = {'LayerType': 'CNN', 'LayerName': 'M4L28', 'M': 512, 'N': 2048, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L29 = {'LayerType': 'CNN', 'LayerName': 'M4L29', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M4L30 = {'LayerType': 'CNN', 'LayerName': 'M4L30', 'M': 2048, 'N': 512, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L31 = {'LayerType': 'CNN', 'LayerName': 'M4L31', 'M': 512, 'N': 2048, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L32 = {'LayerType': 'CNN', 'LayerName': 'M4L32', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M4L33 = {'LayerType': 'CNN', 'LayerName': 'M4L33', 'M': 2048, 'N': 512, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L34 = {'LayerType': 'CNN', 'LayerName': 'M4L34', 'M': 512, 'N': 2048, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L35 = {'LayerType': 'CNN', 'LayerName': 'M4L35', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M4L36 = {'LayerType': 'CNN', 'LayerName': 'M4L36', 'M': 2048, 'N': 512, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L37 = {'LayerType': 'CNN', 'LayerName': 'M4L37', 'M': 512, 'N': 2048, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}
M4L38 = {'LayerType': 'CNN', 'LayerName': 'M4L38', 'M': 512, 'N': 512, 'R': 28, 'C': 28, 'K': 3, 'S': 1, 'P': 1}
M4L39 = {'LayerType': 'CNN', 'LayerName': 'M4L39', 'M': 2048, 'N': 512, 'R': 28, 'C': 28, 'K': 1, 'S': 1, 'P': 1}





M4 = {'M4L1':  M4L1,  'M4L2':  M4L2,  'M4L3':  M4L3,  'M4L4':  M4L4,  'M4L5':  M4L5
    ,
      'M4L6':  M4L6
    ,  'M4L7':  M4L7,  'M4L8':  M4L8,  'M4L9':  M4L9,  'M4L10': M4L10,
      'M4L11': M4L11, 'M4L12': M4L12, 'M4L13':  M4L13, 'M4L14': M4L14, 'M4L15':  M4L15,
      'M4L16': M4L16, 'M4L17': M4L17, 'M4L18': M4L18, 'M4L19': M4L19, 'M4L20': M4L20
      }



Modality7 = [ M3, M4]
# Modality7 = [ M3]
# InitMapper.Gmap.add_edge('M1L3', 'M3L7')
# InitMapper.Gmap.add_edge('M2L3', 'M3L7')
if __name__ == "__main__":
    import ParaCount
    paraNum = ParaCount.paraCount(Modality7)
    print(paraNum/1000000)