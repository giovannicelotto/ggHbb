# %%
import glob
import pandas as pd
import sys

#m = int(sys.argv[1])



m=125
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/%d/*.parquet"%m)
fileNames = sorted(fileNames, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
fileNames = fileNames[:20] if m==125 else fileNames
print(fileNames)
df = pd.read_parquet(fileNames)
# %%

# Function to create rows for each pair of jets
def create_jet_pairs(df):
    pairs = []
    for idx, row in df.iterrows():
        true_pair = sorted([row['true1'], row['true2']])  # Get the true pair for comparison
        
        jet_pairs = [
            # Jet 1 and Jet 2
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_nTrigMuons',
                  'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_nTrigMuons']].tolist(), 
             (1 if true_pair == [0, 1] else 0)),  # True pair check
            # Jet 1 and Jet 3
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_nTrigMuons',
                  'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_nTrigMuons']].tolist(),
             (1 if true_pair == [0, 2] else 0)),
            # Jet 1 and Jet 4
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [0, 3] else 0)),
            # Jet 2 and Jet 3
            (row[['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_nTrigMuons',
                  'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_nTrigMuons']].tolist(),
             (1 if true_pair == [1, 2] else 0)),
            # Jet 2 and Jet 4
            (row[['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [1, 3] else 0)),
            # Jet 3 and Jet 4
            (row[['jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [2, 3] else 0))


        ]
        
        # Append each pair's features and its corresponding label
        for features, label in jet_pairs:
            pairs.append(features + [label])  # Add the binary label

    return pairs

# Apply the function to your dataframe
jet_pairs = create_jet_pairs(df)

# Create a new dataframe with jet pairs
columns = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_nTrigMuons',
           'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_nTrigMuons',
           'is_true_pair']  # Add a column for the binary label

pairs_df = pd.DataFrame(jet_pairs, columns=columns)
pairs_df['massHypo']=m
pairs_df.to_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs/%d.parquet"%m)

# View the new dataframe
print(len(pairs_df), " events for M=%d"%m)

'''
m = int(sys.argv[1]) mass of the spin 0 particle [50, 70, 100, 200, 300]
'''
