import numpy as np
import uproot
import sys
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
def findDaughters(myGenParts, motherIdx, GenPart_genPartIdxMother):
    '''Given a b quark idx (motherIdx) find all the daughters.'''
    daughters=[]
    for i in myGenParts:
        if GenPart_genPartIdxMother[i] == motherIdx:
            daughters.append(int(i))
    daughters = np.array(daughters)
    return daughters

def print_values_and_occurrences(arr, maxEntries):
    unique_values, counts = np.unique(arr, return_counts=True)

    for value, count in zip(unique_values, counts):
        print("%d: %.1f "%(value, count/maxEntries*100))

def main():
    filePath  = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/0000/ggH_Run2_mc_2023Dec06_15.root"
    #filePath  = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/ggH_noTrig_Run2_mc_124X.root"
    f = uproot.open(filePath)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    print("Entries : %d"%(maxEntries))

    totalGenMuons = 0
    totalRecoMuons = 0
    higgsEndingWithMuons = 0
    muonsFromB = 0
    muonsFromTau = 0
    muonsFromD = 0
    muonsFromMuons = 0
    muonsFromPi = 0
    muonsFromKaons = 0

    motherMuons = []
    isTriggering=[]
    triggeredByMatched = 0

    for ev in  range(maxEntries):
        #sys.stdout.write('\r')
        #sys.stdout.write("%.1f%%  %.1f%%  %d%%  %.1f%%  %.1f%%  %.1f%%  %.1f%%  "%(higgsEndingWithMuons/(ev+1)*100, muonsFromB/(totalGenMuons+1)*100, muonsFromD/(totalGenMuons+1)*100, muonsFromTau/(totalGenMuons+1)*100, muonsFromMuons/(totalGenMuons+1)*100, muonsFromPi/(totalGenMuons+1)*100, muonsFromKaons/(totalGenMuons+1)*100))
        #sys.stdout.flush()
        nGenPart                    = branches['nGenPart'][ev]
        GenPart_pdgId               = branches['GenPart_pdgId'][ev]
        GenPart_statusFlags         = branches['GenPart_statusFlags'][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]
        Muon_genPartIdx             = branches["Muon_genPartIdx"][ev]
        nMuon                       = branches["nMuon"][ev]
        GenPart_genPartIdxMother    = branches["GenPart_genPartIdxMother"][ev]

        myGenParts = np.arange(0, nGenPart)
        
        
        myGenHiggsMask = (GenPart_pdgId == 25) & (GenPart_statusFlags>=8192)
        myGenHiggs = myGenParts[myGenHiggsMask]
        myGenMuonMask = (abs(GenPart_pdgId) == 13)
        myGenMuons = myGenParts[myGenMuonMask]
        motherMuons=np.append(motherMuons, GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]])
        
        totalGenMuons=totalGenMuons+len(myGenMuons)
        totalRecoMuons = totalRecoMuons+nMuon

# h-> ... -> mu
        if len(myGenMuons)>0:
            higgsEndingWithMuons=higgsEndingWithMuons+1

# h -> ... -> B -> ... -> mu
        #muonsFromB=muonsFromB + np.sum(np.isin(abs(GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]]), [511, 521, 513, 523]))
        #muonsFromTau=muonsFromTau + np.sum(abs(GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]]) == 15)
        #muonsFromD=muonsFromD + np.sum(np.isin(abs(GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]]), [411, 421, 423, 415, 425, 431, 433, 435]))
        #muonsFromPi=muonsFromPi + np.sum(np.isin(abs(GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]]), [111, 211]))
        #muonsFromKaons=muonsFromKaons + np.sum(np.isin(abs(GenPart_pdgId[GenPart_genPartIdxMother[myGenMuons]]), [130, 310, 311, 321, 313, 325]))


        if np.sum(Muon_isTriggering[Muon_genPartIdx>-1])>0:
            triggeredByMatched=triggeredByMatched+1
        
        for genMuIdx in myGenMuons:
            motherPdgId = 999
            flag=False
            while motherPdgId!=25:
                assert GenPart_genPartIdxMother[genMuIdx]!=-1
                motherPdgId=GenPart_pdgId[GenPart_genPartIdxMother[genMuIdx]]
                genMuIdx = GenPart_genPartIdxMother[genMuIdx]
                if (abs(motherPdgId) in [511, 521]):
                    flag=True
            if flag:
                pass
                #higgsEndingWithMuons=higgsEndingWithMuons+1
                #print(motherPdgId)
        
        for genMuIdx in myGenMuons:
            for recoMuIdx in range(nMuon):
                if genMuIdx==Muon_genPartIdx[recoMuIdx]:
                    isTriggering.append(Muon_isTriggering[recoMuIdx])

            
            
            # Check that all the gen muons comes from Higgs. Of course i retained aonly genpart in higgs decay chain
            #motherPdgId = 999
            #input("Next : %d, %d"%(ev, muIdx))
            #while motherPdgId!=25:
            #    assert GenPart_genPartIdxMother[muIdx]!=-1
            #    motherPdgId=GenPart_pdgId[GenPart_genPartIdxMother[muIdx]]
            #    muIdx = GenPart_genPartIdxMother[muIdx]
            #    print(motherPdgId)
        #print(ev/maxEntries*100, np.sum(isTriggering)/(len(isTriggering)))
        assert(np.array(GenPart_statusFlags[myGenHiggs]<16382).all())               # check isLastCopyBeforeFSR (16384) and isLastCopy (8192)



    
    print("Triggering", np.sum(isTriggering))
    print("matched muons", len(isTriggering))
    print("total reco ", totalRecoMuons)
    print("total gen ", totalGenMuons)
    print("mother muons", len(motherMuons))
    print("Events where at least one triggered muon comes from a matched one", triggeredByMatched)


    print_values_and_occurrences(motherMuons, len(motherMuons))
    unique_values, counts = np.unique(abs(motherMuons), return_counts=True)
    counts=counts/len(motherMuons)
    bins = np.arange(0, len(unique_values)+1)
    fig, ax = plt.subplots(1, 1, figsize=(30, 6))
    ax.hist(bins[:-1], bins=bins-0.5, weights=counts, color='blue', histtype=u'step')
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(i) for i in unique_values], rotation=90)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("PDG ID")
    ax.set_ylabel("Probability")
    ax.set_title("Pdg ID of mothers of muons")
    for i, count in enumerate(counts):
        ax.text(bins[i], counts[i]*1.05, "%.1f%%"%(count*100), ha='center', va='bottom', color='black')

    fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/motherGenMuons.png", bbox_inches='tight')
if __name__ == "__main__":
    main()
