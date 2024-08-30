import numpy as np
import glob
import pandas as pd

pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
fileNames = glob.glob(pathToPredictions)
fileNumberList=[]
processNumberList=[]
for fileName in fileNames:
    match = re.search(r'fn(\d+)_pt%d'%pTClass, fileName)
