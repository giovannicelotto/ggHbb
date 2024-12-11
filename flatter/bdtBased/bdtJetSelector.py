from itertools import combinations
import pandas as pd
import xgboost as xgb
import random
def bdtJetSelector(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_btagDeepFlavB, Jet_nTrigMuons, bst_loaded):

    # Generate all pairs of jets using itertools.combinations
    jet_indices = list(combinations(range(len(Jet_pt)), 2))

    # Create a list of dictionaries to store the combined jet pairs
    rows = []
    for (i, j) in jet_indices:
        row = {
            'jet1_pt': Jet_pt[i], 'jet1_eta': Jet_eta[i], 'jet1_phi': Jet_phi[i], 'jet1_mass': Jet_mass[i],
            'jet1_btagDeepFlavB': Jet_btagDeepFlavB[i], 'jet1_nTrigMuons': Jet_nTrigMuons[i],
            
            'jet2_pt': Jet_pt[j], 'jet2_eta': Jet_eta[j], 'jet2_phi': Jet_phi[j], 'jet2_mass': Jet_mass[j],
            'jet2_btagDeepFlavB': Jet_btagDeepFlavB[j], 'jet2_nTrigMuons': Jet_nTrigMuons[j],
            #'massHypo':125,
            'jet1_index':i, 
            'jet2_index':j,
            'is_true_pair': 1 if [i, j] == [0, 1] else 0  # Example: Mark true pair (optional, modify as needed)
        }
        rows.append(row)

    # Create a DataFrame from the combined rows
    pair_df = pd.DataFrame(rows)
    #pair_df['massHypo']=125 if isMC==1 else random.choice([50, 70, 100, 125, 200, 300])


    feature_columns = [
        'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_nTrigMuons',
        'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_nTrigMuons',
        #'massHypo'
    ]

    # Extract the feature DataFrame
    X = pair_df[feature_columns]
    dtest = xgb.DMatrix(X)

    # Make predictions using the BDT model
    bdt_predictions = bst_loaded.predict(dtest)

    # Add the predictions as a new column in the DataFrame
    pair_df['bdt_score'] = bdt_predictions


    max_bdt_index = pair_df['bdt_score'].idxmax()

    # Get the jet1_index and jet2_index from the row with the highest BDT score
    jet1_index = pair_df.loc[max_bdt_index, 'jet1_index']
    jet2_index = pair_df.loc[max_bdt_index, 'jet2_index']

    return jet1_index, jet2_index