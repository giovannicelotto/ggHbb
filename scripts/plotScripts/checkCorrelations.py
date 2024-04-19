import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, sys


print("Program Starting")
flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"

flatFileNames = glob.glob(flatPathCommon+"/QCD_Pt600To800"+"/**/*.parquet", recursive=True)
outName = "/t3home/gcelotto/ggHbb/outputs/plots/features/corr_QCD600To800.pdf"
#dataFolder = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/others/*.parquet'
#dataFolder = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others/*.parquet'
#fileNames = glob.glob(dataFolder)[:2]
print(flatFileNames)
#sys.exit("Exit")
df = pd.read_parquet(flatFileNames, columns=['dijet_mass', 'dijet_twist', 'jet1_btagDeepFlavB'])


correlation_matrix = df.corr()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8}, cmap="coolwarm",
            mask=~((abs(correlation_matrix) > 0.0) | (correlation_matrix < -0.5)))

ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_yticklabels(), rotation=90)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
ax.set_title("Correlation matrix QCD600To800")
fig.savefig(outName, bbox_inches='tight')