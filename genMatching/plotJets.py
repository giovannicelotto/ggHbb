import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import mplhep as hep
hep.style.use("CMS")

fig,ax = plt.subplots(2, 2)
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched/GluGlu*.parquet")
df = pd.read_parquet(fileNames)

m = (df.jet1_pt > 20)  & (abs(df.jet1_eta)<2.5) & (abs(df.jet1_eta)<2.5)
print(len(df), " entries")
print(len(df[m]), " in the phase space")
m=(m)& (df.genPart1_pt>1e-5)
print(len(df[m]), " with matching")
ax[0,0].hist(df.jet1_pt[m] - df.genPart1_pt[m],     bins = np.linspace(-100,100,50), alpha=0.5, label='Jet with TrigMuon')
ax[0,1].hist(df.jet1_eta[m] - df.genPart1_eta[m],   bins = np.linspace(-2.5,2.5,50), alpha=0.5, label='Jet with TrigMuon')
ax[1,0].hist(df.jet1_phi[m] - df.genPart1_phi[m],   bins = np.linspace(-2*np.pi,2*np.pi,50), alpha=0.5, label='Jet with TrigMuon')
ax[1,1].hist(df.jet1_mass[m] - df.genPart1_mass[m], bins = np.linspace(-50,50,50), alpha=0.5, label='Jet with TrigMuon')



print(df.head(10))
for n, ev in enumerate(df):
    print(df.jet1_phi[ev], df.genPart1_phi[ev])
    if n%100==0:
        letter=input("Continue:")
        if letter=='y':
            pass
        else:
            break
#ax.set_xlabel("Jet pT - Quark pT [GeV]")
#ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/genMatching/JetVsQuark.png", bbox_inches='tight')
