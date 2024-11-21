def getZdf(dfs, isMCList):
    dfZ = []
    for idx,df in enumerate(dfs):
        if (isMCList[idx] == 2) | (isMCList[idx] == 20) | (isMCList[idx] == 21) | (isMCList[idx] == 22) | (isMCList[idx] == 23) | (isMCList[idx] == 36):
            dfZ.append(df)
    dfZ=pd.concat(dfZ)
    return dfZ