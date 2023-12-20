import ROOT
import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
rootPath = "/t3home/gcelotto/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test/Hbb_noTrig_Run2_mc_124X.root"
f = uproot.open(rootPath)
tree = f['Events']
branches = tree.arrays()
maxEntries = 1000#tree.num_entries
print("Entries : %d"%(maxEntries))

totalEntries = 0

for ev in  range(maxEntries):
    sys.stdout.write('\r')
    sys.stdout.write("%d%%"%(ev/maxEntries*100))
    sys.stdout.flush()
    nGenPart        = branches['nGenPart'][ev]
    GenPart_pdgId   = branches['GenPart_pdgId'][ev]
    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
    GenPart_pt = branches["GenPart_pt"][ev]

    
    myGenParts = np.arange(0, nGenPart)
    if ((nGenPart>0) & (np.sum(GenPart_pdgId==25)>=1)):
        totalEntries=totalEntries+1

    myGenMuons = myGenParts[abs(GenPart_pdgId)==13]
    myGenBMesons = [i for i in myGenParts if GenPart_pdgId[i] in [521, -521, 511, -511]]
    #if len(myGenBMesons)>0:
    #    print(ev,GenPart_pdgId[myGenBMesons])
    for mu in myGenMuons:
        motherMu = GenPart_genPartIdxMother[mu]
        if GenPart_pdgId[motherMu] not in [521, -521, 511, -511]:
            continue
        if GenPart_pdgId[GenPart_genPartIdxMother[motherMu]] not in [-5, 5]:
            if GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[motherMu]]] not in [-5, 5]:
                continue
        if GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[motherMu]]] not in [25]:
            if GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[GenPart_genPartIdxMother[motherMu]]]] not in [25]:
                continue
        muFromHiggs=muFromHiggs+1

#TwoBMesEvent = 0
#for bMes in myGenBMesons:
#    motherBMes = GenPart_genPartIdxMother[bMes]
#    if GenPart_pdgId[motherBMes] not in [-5, 5]:
#        continue
#    if GenPart_pdgId[GenPart_genPartIdxMother[motherBMes]] not in [25]:
#        continue
#    TwoBMesEvent=TwoBMesEvent+1
#if TwoBMesEvent==2:
#    BMesFromHiggs=BMesFromHiggs+1
#    pass
#else:
#    continue


#print("MuFromHiggs ", muFromHiggs, muFromHiggs/totalEntries)
#print("BMesFromHiggs", BMesFromHiggs, BMesFromHiggs/totalEntries)