import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
def print_values_and_occurrences(arr, maxEntries):
    unique_values, counts = np.unique(arr, return_counts=True)

    for value, count in zip(unique_values, counts):
        print("%d: %.1f "%(value, count/maxEntries*100))


def findDaughters(myGenParts, motherIdx, GenPart_genPartIdxMother):
    '''Given a b quark idx (motherIdx) find all the daughters.'''
    daughters=[]
    for i in myGenParts:
        if GenPart_genPartIdxMother[i] == motherIdx:
            daughters.append(i)
    daughters = np.array(daughters)
    return daughters

def findDaughtersExcept(myGenParts, vetoDaughters, motherIdx, GenPart_genPartIdxMother, GenPart_pdgId):
    '''Given a b quark idx (motherIdx) find all the daughters.
    if a daughter is still a bquark replace it with the next daughter until no b quarks are found in the list'''
    daughters=findDaughters(myGenParts, motherIdx, GenPart_genPartIdxMother)
    # daughters is a list of daughters of the b quark of interest. But it might contain other b quarks
    if len(daughters)>0:
        #print("Daughters", GenPart_pdgId[daughters])
        while any(x in GenPart_pdgId[daughters] for x in vetoDaughters):#(5 in GenPart_pdgId[daughters]) | (-5 in GenPart_pdgId[daughters]) | (21 in GenPart_pdgId[daughters]): 
            #print("Found b", daughters, GenPart_pdgId[daughters])
            # until when there is a b (anti)quark in the list of ultimate daughters:
            for i in daughters[np.isin(GenPart_pdgId[daughters], vetoDaughters)]:
                # take these b quarks in the dughters list
                # i is a daughter of the original b quark which is in turn a b quark
                newDaughters = findDaughters(myGenParts, i, GenPart_genPartIdxMother)
                daughters= daughters[daughters != i]
                if len(newDaughters)==0:
                    break

                # remove the b quark from the daughters of the b quark
                daughters=np.append(daughters, newDaughters)
    else:
        pass#print("no b quarks")

    return daughters

def main():
    '''
    1. categorize daughters of Bmesons different (511, 521) different from themselves.
    2. count b quarks daughters of higgs and histogram the number of b quarks from higgs'''
    rootPath = "/t3home/gcelotto/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test/Hbb_noTrig_Run2_mc_124X.root"
    f = uproot.open(rootPath)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = 5000#tree.num_entries
    print("Entries : %d"%(maxEntries))

    totalEntries = 0                            # count number of times nGenPart>0 and nHiggs>0
    totalBMesons = 0                            # sum event by event the length of myGenBMesons (array of particles with pdgID=521, 511)
    histBQuarks = []                            # event by event append the length of myGenBQuarks (array of (5, -5) excluding daughters of 5, -5)
    daughtersBQuarks = np.array([])             # array of pdgId of daughters of b quarks
    daughtersOfBstar_zero = np.array([])        # array of pdgId of daughters of Bstar_zero
    daughtersOfBstar_charged = np.array([])     # array of pdgId of daughters of Bstar_charged
    daughtersOfBMesons = np.array([])           # array of pdgId of daughters of BMesons
    for ev in  range(maxEntries):
        sys.stdout.write('\r')
                # the exact output you're looking for:
        sys.stdout.write("%d%%"%(ev/maxEntries*100))
        sys.stdout.flush()
        nGenPart        = branches['nGenPart'][ev]
        GenPart_pdgId   = branches['GenPart_pdgId'][ev]
        GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
        #Muon_isTriggering = branches["Muon_isTriggering"][ev]

        myGenParts = np.arange(0, nGenPart)
        if ((nGenPart>0) & (np.sum(GenPart_pdgId==25)>=1)):
            totalEntries=totalEntries+1
        myGenBQuarks = [i for i in myGenParts if (GenPart_pdgId[i] in [5, -5]) & (GenPart_pdgId[GenPart_genPartIdxMother[i]]==25)]
        myGenBExcitedMesons_zero = [i for i in myGenParts if (GenPart_pdgId[i] in [513, -513]) & (abs(GenPart_pdgId[GenPart_genPartIdxMother[i]])==5)]
        myGenBExcitedMesons_charged = [i for i in myGenParts if (GenPart_pdgId[i] in [523, -523]) & (abs(GenPart_pdgId[GenPart_genPartIdxMother[i]])==5)]
        myGenBMesons = [i for i in myGenParts if (GenPart_pdgId[i] in [521, -521, 511, -511])]

        if len(myGenBMesons)>0:
            for mesIdx in myGenBMesons:
                if GenPart_genPartIdxMother[mesIdx] in myGenBMesons:
                    assert False
        totalBMesons+=len(myGenBMesons)
        
        for motherIdx in myGenBQuarks:    
            daughtersToAppend = findDaughtersExcept(myGenParts,[5, -5, 21], motherIdx, GenPart_genPartIdxMother, GenPart_pdgId)
            if len(daughtersToAppend)>0:
                daughtersToAppend=GenPart_pdgId[daughtersToAppend]
                daughtersBQuarks = np.append(daughtersBQuarks, daughtersToAppend)
        
        for motherIdx in myGenBExcitedMesons_zero:    
            daughtersToAppend = findDaughtersExcept(myGenParts, [513, -513], motherIdx, GenPart_genPartIdxMother, GenPart_pdgId)
            if len(daughtersToAppend)>0:
                daughtersToAppend=GenPart_pdgId[daughtersToAppend]
                daughtersOfBstar_zero = np.append(daughtersOfBstar_zero, daughtersToAppend)
                
        for motherIdx in myGenBExcitedMesons_charged:    
            daughtersToAppend = findDaughtersExcept(myGenParts,[523, -523], motherIdx, GenPart_genPartIdxMother, GenPart_pdgId)
            if len(daughtersToAppend)>0:
                daughtersToAppend=GenPart_pdgId[daughtersToAppend]
                daughtersOfBstar_charged = np.append(daughtersOfBstar_charged, daughtersToAppend)
        
        for motherIdx in myGenBQuarks:    
            daughtersToAppend = findDaughtersExcept(myGenParts, [511, -511, 521, -521], motherIdx, GenPart_genPartIdxMother, GenPart_pdgId)
            if len(daughtersToAppend)>0:
                daughtersToAppend=GenPart_pdgId[daughtersToAppend]
                daughtersOfBMesons = np.append(daughtersOfBMesons, daughtersToAppend)
    
        histBQuarks.append(len(myGenBQuarks))
        #if (len(myGenBQuarks)==1) & (np.sum(Muon_isTriggering)>0):
        #    print(ev)


    c, b = np.histogram(histBQuarks, bins=np.arange(0, 6))[:2]
    print("totalEntries nGenPart>0 & nHiggs>0", totalEntries, totalEntries/maxEntries)
    for num, count in zip(b, c):
        print("%d b quarks : %d %.1f"%(num, count, count/maxEntries*100))
    print_values_and_occurrences(daughtersBQuarks, len(daughtersBQuarks))
    print("Excited B mesons decay neutral:")
    print_values_and_occurrences(daughtersOfBstar_zero, len(daughtersOfBstar_zero))
    
    print("\n\nExcited B mesons decay charged:")
    print_values_and_occurrences(daughtersOfBstar_charged, len(daughtersOfBstar_charged))

    print("\n\nB mesons (511, 521) daughters:")
    print_values_and_occurrences(abs(daughtersOfBMesons), totalBMesons)

    unique_values, counts = np.unique(abs(daughtersOfBMesons), return_counts=True)
    counts=counts/totalBMesons
    bins = np.arange(0, len(unique_values)+1)
    fig, ax = plt.subplots(1, 1, figsize=(30, 6))
    ax.hist(bins[:-1], bins=bins-0.5, weights=counts, color='blue', histtype=u'step')
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(i) for i in unique_values], rotation=90)
    ax.set_xlabel("PDG ID")
    ax.set_ylabel("Probability")
    ax.set_title("Pdg ID of daughters of B Mesons")
    #ax.set_xscale('log')
    fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/pdgIdDauhtersFromBMesons.png", bbox_inches='tight')

     
if __name__=="__main__":
    main()























#rootFile = ROOT.TFile(rootPath)
#
## Check if the file is successfully opened
#if not rootFile.IsOpen():
#    print(f"Error: Unable to open file {rootPath}")
#    exit()
#
#tree_name = "Events"
#tree = rootFile.Get(tree_name)
#
## Check if the tree is successfully retrieved
#if tree is None:
#    print(f"Error: Unable to retrieve tree {tree_name} from file {rootPath}")
#    rootFile.Close()
#    exit()
#
## Create a canvas for drawing
#canvas = ROOT.TCanvas("canvas", "My Canvas", 800, 600)
#ROOT.gStyle.SetOptStat(0)
#
#h1 = ROOT.TH1F("h1", "Muon pT;;", 100, 0, 100)
#h2 = ROOT.TH1F("h2", "TrigObj pT;;", 100, 0, 100)
#h3 = ROOT.TH1F("h3", "GenMuon pT;;", 100, 0, 100)
#
#tree.Draw("Muon_pt[0]>>h1", "")
#tree.Draw("TrigObj_pt[0]>>h2", "", "same")
#tree.Draw("GenPart_pt>>h3", "(abs(GenPart_pdgId)==13) & (GenPart_pdgId[GenPart_genPartIdxMother]==521)")
#h2.SetLineColor(ROOT.kRed)
#max_y = max(h1.GetMaximum(), h2.GetMaximum())
#h1.GetYaxis().SetRangeUser(0, 1.2 * max_y)
#canvas.GetPad(0).Update()
#canvas.GetPad(0).Modified()
#
#
#legend = ROOT.TLegend(0.7, 0.75, 0.90, 0.87)
#legend.AddEntry(h1, h1.GetTitle(), "l")
#legend.AddEntry(h2, h2.GetTitle(), "l")
#legend.SetBorderSize(0)
#
#canvas.cd()
#legend.Draw()
#canvas.Draw()
#canvas.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/Muon_pt_miniAOD.pdf")
#rootFile.Close()

'''print("Entries : %d"%(maxEntries))
for ev in  range(maxEntries):
    nGenPart        = branches['nGenPart'][ev]
    GenPart_pdgId   = branches['GenPart_pdgId'][ev]
    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
    GenPart_pt = branches["GenPart_pt"][ev]

    
    myGenParts = np.arange(0, nGenPart)
    myGenMuons = myGenParts[abs(GenPart_pdgId)==13]
    if len(myGenMuons)>0:
        input("Next")
    for mu in myGenMuons:
        # daughters of Bmesons
        HasBMesonMother = False
        motherIdx = GenPart_genPartIdxMother[mu]  #take the mother idx of the muon
        motherDegree = 0
        while GenPart_pdgId[motherIdx]!=2212:
            motherPdg = GenPart_pdgId[motherIdx]
            print(ev, mu, motherPdg)'''