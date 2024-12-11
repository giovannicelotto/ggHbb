

def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId):
    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
    muonIdx2 = 999
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in range(nJet): 
        if (abs(Jet_eta[i])>2.5) | (Jet_puId[i]<4) | (Jet_jetId[i]<6):     # exclude jets>2.5 from the jets with  muon group
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
    # if two jets
    if len(muonIdxs)>=2:
        selected1=jetsWithMuon[0]
        selected2=jetsWithMuon[1]
        muonIdx=muonIdxs[0]
        muonIdx2=muonIdxs[1]
    elif len(muonIdxs)==0:
        selected1=999
        selected2=999
        muonIdx=999
        muonIdx2=999
    elif len(muonIdxs)==1:
        selected1 = jetsWithMuon[0]
        muonIdx = muonIdxs[0]
        for j in range(0, jetsToCheck):
            if j==jetsWithMuon[0]:
                continue
            if (abs(Jet_eta[j])>2.5) | (Jet_puId[j]<4) | (Jet_jetId[j]<6):
                continue
            currentScore = Jet_btagDeepFlavB[j]
            if currentScore>score:
                score=currentScore
                selected2 = j
                muonIdx2 = 999
        if selected2 == 999: # case there are not 2 jets in the acceptance set also the first to 999 to say there is no pair chosen
            selected1 = 999
    else:
        assert False
    return selected1, selected2, muonIdx, muonIdx2