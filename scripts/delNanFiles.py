import pandas as pd
import glob, os

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/"
fileNames = glob.glob(path+"*/*.parquet")

for fileName in fileNames:
    
    df = pd.read_parquet(fileName)
    
    if (df.isna().sum().jet2_qgl==1):
        print(fileName)
        os.remove(fileName)