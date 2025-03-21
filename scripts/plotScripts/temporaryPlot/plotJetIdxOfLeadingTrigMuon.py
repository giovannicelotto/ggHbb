import uproot
import ROOT
import glob
''' 
    script that takes root nanoAOD files of BPH2018 1A and save the index of the jet that contains the leading triggering muon in a histogram
    if there are no triggering muons inside jets save -1
    if there are no muon that are triggering (punch-through e.g.) fills with -3
'''
signal = True
dataPathBkg = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/0000"
dataPathSignal = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/0000"
dataPath = dataPathSignal if signal else dataPathBkg
fileNames = glob.glob(dataPath + "/*.root")

canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)
histogram = ROOT.TH1F("histogram", "", 35, -5, 30)
histogram.GetXaxis().SetTitle("JetIdx containing Leading Triggering Muon")
histogram.GetYaxis().SetTitle("Events")


totalEntries=0
for fileName in fileNames[0:10]:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    totalEntries+=maxEntries

    print("%d/%d"%(fileNames.index(fileName)+1, len(fileNames)), fileName, "\n\nEntries : %d"%maxEntries)
    for ev in  range(maxEntries):
        filled=False
        #Muon_pt                     = branches["Muon_pt"][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]
        nJet                        = branches["nJet"][ev]
        nMuon                       = branches["nMuon"][ev]
        Jet_muonIdx1                = branches["Jet_muonIdx1"][ev]
        nTrigObj                    = branches["nTrigObj"][ev]
        #Muon_fired_HLT_Mu9_IP6      = branches["Muon_fired_HLT_Mu9_IP6"][ev]
        if ((sum(Muon_isTriggering)==0) & (nTrigObj>0)):
            
            histogram.Fill(-3) # noMuon triggering (trig objects was identified as something else)
            filled=True
            continue
        for mu in range(nMuon):
            #if Muon_isTriggering[mu]==1:
            if Muon_isTriggering[mu]==1:
                foundJet=False
                for idx in range(nJet):
                    if Jet_muonIdx1[idx]==mu:
                        histogram.Fill(idx)
                        filled=True
                        foundJet=True
                        break
                if foundJet:
                    break
                if foundJet==False:
                    filled=True
                    histogram.Fill(-1)  # no Muons inside jets
                    break
        assert filled, "Entry not filled"

            
            
histogram.Scale(1./histogram.GetEntries())
histogram.Draw("histtext")
canvas.SetLogy()
canvas.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/JetIdx_Signal_containing_Leading_Triggering_Muon_SignalEvents.pdf")
print(histogram.Integral(-1, 35))
print("Total Scanned Entries : %d"%totalEntries)


