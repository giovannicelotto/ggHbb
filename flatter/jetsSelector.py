import numpy as np

def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, method=0, Jet_pt=None):
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
    muonIdx2 = 999
# Fill the list of Jets with TrigMuon inside
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if ((abs(Jet_eta[i])>2.5) | ((Jet_pt[i]<50) & (Jet_puId[i]<4)) | (Jet_jetId[i]<6)):     # exclude jets>2.5 from the jets with  muon group
            continue
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if (bool(Muon_isTriggering[Jet_muonIdx1[i]])):
                if i not in jetsWithMuon:
                    jetsWithMuon.append(i)
                    muonIdxs.append(Jet_muonIdx1[i])
                    continue
        if (Jet_muonIdx2[i]>-1):
            if (bool(Muon_isTriggering[Jet_muonIdx2[i]])):
                if i not in jetsWithMuon:
                    jetsWithMuon.append(i)
                    muonIdxs.append(Jet_muonIdx2[i])
                    continue
    assert len(muonIdxs)==len(jetsWithMuon)
# Now loop over these jets as first element of the pair
    # 2+ Jets With Trig
    if len(muonIdxs)>=2:
        selected1=jetsWithMuon[0]
        selected2=jetsWithMuon[1]
        muonIdx=muonIdxs[0]
        muonIdx2=muonIdxs[1]
    # No Jets With Trig Muon -> idxs are set to 999 (events to be rejected)
    elif len(muonIdxs)==0:
        selected1=999
        selected2=999
        muonIdx=999
        muonIdx2=999
    # 1 Jet. Choose the second in a way
    elif len(muonIdxs)==1:
        selected1 = jetsWithMuon[0]
        muonIdx = muonIdxs[0]
        if method==0:
            # old method based on btag shape
            for j in range(0, jetsToCheck):
                if j==jetsWithMuon[0]:
                    continue
                if (abs(Jet_eta[j])>2.5) | ((Jet_pt[j]<50) & (Jet_puId[j]<4)) | (Jet_jetId[j]<6):
                    continue
                currentScore = Jet_btagDeepFlavB[j]
                if currentScore>score:
                    score=currentScore
                    selected2 = j
                    muonIdx2 = 999
        elif method==1:
            # New method. List of Tight BTag jets with puID >=4 and JetID pass tight and tightLepVeto ID.
            tightJets = (Jet_btagDeepFlavB>0.71) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4))
            if np.sum(tightJets)>=1:
                #print(nJet)
                #print(np.arange(nJet)[tightJets])
                selected2 = np.arange(nJet)[tightJets][0]

                pass
                # hai vinto prendi il piú hard
            elif np.sum(tightJets)==0:
                mediumJets = (Jet_btagDeepFlavB>0.2783) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4))
                if np.sum(mediumJets)>=1:
                    selected2 = np.arange(nJet)[mediumJets][0]
                    #haivinto
                    pass
                elif np.sum(mediumJets)==0:
                    looseJets = (Jet_btagDeepFlavB>0.0490) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4))
                    if np.sum(looseJets)>=1:
                        selected2 = np.arange(nJet)[looseJets][0]
                        #hai vinto
                        pass
                    elif np.sum(looseJets)==0:
                        selected2 = 999
                        #haivinto

        
        if selected2 == 999: # case there are not 2 jets in the acceptance set also the first to 999 to say there is no pair chosen
            selected1 = 999
    # No other possibilities
    else:
        assert False
    return selected1, selected2, muonIdx, muonIdx2