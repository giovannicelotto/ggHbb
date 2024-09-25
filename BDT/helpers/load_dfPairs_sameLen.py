import pandas as pd
import glob

def load_dfPairs_sameLen(path):
    dfs = [ ]
    fileNames = glob.glob(path+"/*.parquet")
    min_len = 0
    for fileName in fileNames:
        df = pd.read_parquet(fileName)
        min_len = len(df) if ((len(df)<min_len) | (min_len==0)) else min_len
        dfs.append(df)
    for idx, df in enumerate(dfs):
        dfs[idx] = dfs[idx].head(min_len)
    return dfs, min_len