import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, sys


flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others/"
fileNames = glob.glob(flatPathCommon+"/GluGluHToBB*.parquet")
print(len(fileNames), " found")
df = pd.read_parquet(fileNames)

correlation_matrix = df.corr()

fig, ax = plt.subplots(1, 1, figsize=(25, 25))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8}, cmap="coolwarm",
            mask=~((abs(correlation_matrix) > 0.0) | (correlation_matrix < -0.5)))

    #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    #ax.set_xticklabels(ax.get_yticklabels(), rotation=90)
    #ax.set_xticklabels(df.columns)
    #ax.set_yticklabels(df.columns)
ax.set_title("Correlation matrix GluGluHToBB.pdf")
outName = "/t3home/gcelotto/ggHbb/outputs/plots/features/corr_GluGluHToBB.pdf"
fig.savefig(outName, bbox_inches='tight')
print("Saved ", outName)

#flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets"
#lowerLimit = [100, 200, 400, 600, 800, 'Inf']
#dfs=[]
#for idx, ll in enumerate(lowerLimit[:-1]):
#    print(idx)    
#    fileNames = glob.glob(flatPathCommon+"/ZJetsToQQ_HT-%d*/*.parquet"%ll)
#    df = pd.read_parquet(fileNames)
#    dfs.append(df)
#
#
#
#
#    correlation_matrix = df.corr()
#
#    fig, ax = plt.subplots(1, 1, figsize=(25, 25))
#    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8}, cmap="coolwarm",
#            mask=~((abs(correlation_matrix) > 0.0) | (correlation_matrix < -0.5)))
#
#    #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#    #ax.set_xticklabels(ax.get_yticklabels(), rotation=90)
#    #ax.set_xticklabels(df.columns)
#    #ax.set_yticklabels(df.columns)
#    ax.set_title("Correlation matrix ZJets%dTo%s.pdf"%(ll, str(lowerLimit[idx+1])))
#    outName = "/t3home/gcelotto/ggHbb/outputs/plots/features/corr_ZJets%dTo%s.pdf"%(ll, str(lowerLimit[idx+1]))
#    fig.savefig(outName, bbox_inches='tight')
#    print("Saved ", outName)