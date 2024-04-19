import numpy as np
import glob
import pandas as pd

fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**/*.parquet", recursive=True)
entries = []

for fileName in fileNames:
    print(fileNames.index(fileName), "\r")
    df=pd.read_parquet(fileName)
    entries.append(len(df))

s=np.sum(entries)
m=(np.mean(entries))
std=(np.std(entries))
print("%d +- %d"%(m, std) )
print("sum", s)
