import numpy as np
def jetsSelector_new(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, Muon_pt, Muon_eta, Muon_dxy, Muon_dxyErr, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, maskJets, method=0, Jet_pt=None):
    '''
    selected1 and selected 2 are indexes in the full list of jets not just the masked jets
    '''
    selected1 = 999
    selected2 = 999
    muonIdx = 999
    muonIdx2 = 999
    #List of muons inside good jets
    jetsWithMuon, LeadingMuons_inJets = [], []
    
# Jets With Muon is the list of jets with a muon inside their cone
# LeadingMuons_inJets is the list of indices of muons that are the leading muons inside any jet
    for i in np.arange(nJet)[maskJets]: 
        if ((abs(Jet_eta[i])>2.5) | ((Jet_pt[i]<50) & (Jet_puId[i]<4)) | (Jet_jetId[i]<6)):     # exclude jets>2.5 from the jets with  muon group
            #Masked Jets cannot be here
            assert False
            continue
        
        if (Jet_muonIdx1[i]>-1): #if there is a reco muon in the jet
            if ((Muon_pt[Jet_muonIdx1[i]]>7)  & (abs(Muon_dxy[Jet_muonIdx1[i]] /Muon_dxyErr[Jet_muonIdx1[i]])>3 )):
                LeadingMuons_inJets.append(Jet_muonIdx1[i])
                jetsWithMuon.append(i)
            continue
    
# Now loop over these jets as first element of the pair
    # No jets with muon inside -> Discard the event
    if len(LeadingMuons_inJets)==0:
        # Discard
        selected1=999
        selected2=999
        muonIdx=999
        muonIdx2=999
        # No SF here
    # 1 Jet with muon inside -> Check if triggers. If not discard the event, if yes choose the second jet with b-tag method
    elif len(LeadingMuons_inJets)==1:
        if Muon_isTriggering[LeadingMuons_inJets[0]]:
            selected1 = jetsWithMuon[0]
            muonIdx = LeadingMuons_inJets[0]
            selected2 = -444 # to be chosen
            # SF is effData/effMC of the muon
        else:
            # Discard
            selected1=999
            selected2=999
            muonIdx=999
            muonIdx2=999
            # No SF here
    # Two (or more) jets with muons inside. Consider the first two muons only
    elif len(LeadingMuons_inJets)>=2:
        # Case A) Both are triggering. Jet 1 is the leading
        if ((Muon_isTriggering[LeadingMuons_inJets[0]]) & (Muon_isTriggering[LeadingMuons_inJets[1]])):

            selected1=jetsWithMuon[0] if Muon_pt[LeadingMuons_inJets[0]]>Muon_pt[LeadingMuons_inJets[1]] else jetsWithMuon[1]
            selected2=jetsWithMuon[1] if Muon_pt[LeadingMuons_inJets[0]]>Muon_pt[LeadingMuons_inJets[1]] else jetsWithMuon[0]
            muonIdx=LeadingMuons_inJets[0] if Muon_pt[LeadingMuons_inJets[0]]>Muon_pt[LeadingMuons_inJets[1]] else LeadingMuons_inJets[1]
            muonIdx2=LeadingMuons_inJets[1] if Muon_pt[LeadingMuons_inJets[0]]>Muon_pt[LeadingMuons_inJets[1]] else LeadingMuons_inJets[0]
            # SF is effData1/effMC1 * effData2/effMC2 
        # Case B1) Leading Muon is triggering. Jet 1 is that one. Choose the second jet with b-tag method
        elif ((Muon_isTriggering[LeadingMuons_inJets[0]]) & (not Muon_isTriggering[LeadingMuons_inJets[1]])):
            selected1=jetsWithMuon[0] 
            muonIdx=LeadingMuons_inJets[0]
            selected2=-444
            #selected2 with btag
            # SF is effData1/effMC1 * 1-effData2/1-effMC2 
        elif ((not Muon_isTriggering[LeadingMuons_inJets[0]]) & (Muon_isTriggering[LeadingMuons_inJets[1]])):
            selected1=jetsWithMuon[1] 
            muonIdx=LeadingMuons_inJets[1]
            selected2=-444
            #selected2 with btag
            # SF is effData1/effMC1 * 1-effData2/1-effMC2 
        
        elif ((not Muon_isTriggering[LeadingMuons_inJets[0]]) & (not Muon_isTriggering[LeadingMuons_inJets[1]])):
            selected1=999
            selected2=999
            muonIdx=999
            muonIdx2=999
            # No SF here
    if selected2==-444:
        # Choose second bjet based btag WP
        tightJets = ((Jet_btagDeepFlavB>=0.71) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
        if np.sum(tightJets)>=1:
            #Leading among tight
            selected2 = np.arange(nJet)[maskJets][tightJets][0]

            pass
            # hai vinto prendi il piú hard
        elif np.sum(tightJets)==0:
            mediumJets = ((Jet_btagDeepFlavB>=0.2783) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
            if np.sum(mediumJets)>=1:
                selected2 = np.arange(nJet)[maskJets][mediumJets][0]
                pass
            elif np.sum(mediumJets)==0:
                looseJets = ((Jet_btagDeepFlavB>=0.0490) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
                if np.sum(looseJets)>=1:
                    selected2 = np.arange(nJet)[maskJets][looseJets][0]
                    #hai vinto
                    pass
                elif np.sum(looseJets)==0:
                    nonLooseJets = ((Jet_btagDeepFlavB>=-1) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
                    if np.sum(nonLooseJets)>=1:
                        selected2 = np.arange(nJet)[maskJets][nonLooseJets][0]
                    elif np.sum(nonLooseJets)==0:
                        selected2=999
    # TO compute SF use LeadingMuons_inJets
    # if lenght==1 use LeadingMuons_inJets[0]
    # if lenght==2 use Muon_isTriggering[LeadingMuons_inJets[0]] and Muon_isTriggering[LeadingMuons_inJets[1]]  to compute the SF
    if selected2 == 999: # case there are not 2 jets in the acceptance set also the first to 999 to say there is no pair chosen
        selected1 = 999
    return selected1, selected2, muonIdx, muonIdx2, LeadingMuons_inJets



def jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, Muon_pt, Muon_eta, Muon_dxy, Muon_dxyErr, jetsToCheck, Jet_btagDeepFlavB, Jet_puId, Jet_jetId, maskJets, method=0, Jet_pt=None):
    '''
    selected1 and selected 2 are indexes in the full list of jets not just the masked jets
    '''


    score=-999
    selected1 = 999
    selected2 = 999
    muonIdx = 999
    muonIdx2 = 999
# Fill the list of Jets with TrigMuon inside
# Jets With Muon is the list of jets with a muon that triggers inside their cone
    jetsWithMuon, muonIdxs = [], []
    for i in np.arange(nJet)[maskJets]: 
        if ((abs(Jet_eta[i])>2.5) | ((Jet_pt[i]<50) & (Jet_puId[i]<4)) | (Jet_jetId[i]<6)):     # exclude jets>2.5 from the jets with  muon group
            #Masked Jets cannot be here
            assert False
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
        selected1=jetsWithMuon[0] if Muon_pt[muonIdxs[0]]>Muon_pt[muonIdxs[1]] else jetsWithMuon[1]
        selected2=jetsWithMuon[1] if Muon_pt[muonIdxs[0]]>Muon_pt[muonIdxs[1]] else jetsWithMuon[0]
        muonIdx=muonIdxs[0] if Muon_pt[muonIdxs[0]]>Muon_pt[muonIdxs[1]] else muonIdxs[1]
        muonIdx2=muonIdxs[1] if Muon_pt[muonIdxs[0]]>Muon_pt[muonIdxs[1]] else muonIdxs[0]
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
        # 
        # METHOD 0
        #         
        if method==0:
            # old method based on btag shape
            assert False
            #for j in range(0, jetsToCheck):
            #    if j==jetsWithMuon[0]:
            #        continue
            #    if (abs(Jet_eta[j])>2.5) | ((Jet_pt[j]<50) & (Jet_puId[j]<4)) | (Jet_jetId[j]<6):
            #        continue
            #    currentScore = Jet_btagDeepFlavB[j]
            #    if currentScore>score:
            #        score=currentScore
            #        selected2 = j
            #        muonIdx2 = 999

        # 
        # METHOD 1
        #   

        elif method==1:
            # New method. List of Tight BTag jets with puID >=4 and JetID pass tight and tightLepVeto ID.
            tightJets = ((Jet_btagDeepFlavB>=0.71) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
            if np.sum(tightJets)>=1:
                #Leading among tight
                selected2 = np.arange(nJet)[maskJets][tightJets][0]

                pass
                # hai vinto prendi il piú hard
            elif np.sum(tightJets)==0:
                mediumJets = ((Jet_btagDeepFlavB>=0.2783) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
                if np.sum(mediumJets)>=1:
                    selected2 = np.arange(nJet)[maskJets][mediumJets][0]
                    pass
                elif np.sum(mediumJets)==0:
                    looseJets = ((Jet_btagDeepFlavB>=0.0490) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
                    if np.sum(looseJets)>=1:
                        selected2 = np.arange(nJet)[maskJets][looseJets][0]
                        #hai vinto
                        pass
                    elif np.sum(looseJets)==0:
                        nonLooseJets = ((Jet_btagDeepFlavB>=-1) & (np.arange(nJet)!=selected1) & (Jet_jetId==6) & ((Jet_pt>=50) | (Jet_puId>=4)))[maskJets]
                        if np.sum(nonLooseJets)>=1:
                            selected2 = np.arange(nJet)[maskJets][nonLooseJets][0]
                        elif np.sum(nonLooseJets)==0:
                            selected2=999
                        #haivinto

        
        if selected2 == 999: # case there are not 2 jets in the acceptance set also the first to 999 to say there is no pair chosen
            selected1 = 999
    # No other possibilities
    else:
        assert False

    return selected1, selected2, muonIdx, muonIdx2