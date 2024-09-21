# %%
import glob
import pandas as pd
# %%

fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/*.parquet")
df = pd.read_parquet(fileNames)
# %%
import pandas as pd

# Function to create rows for each pair of jets
def create_jet_pairs(df):
    pairs = []
    for idx, row in df.iterrows():
        true_pair = sorted([row['true1'], row['true2']])  # Get the true pair for comparison
        
        jet_pairs = [
            # Jet 1 and Jet 2
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_qgl', 'jet1_nTrigMuons',
                  'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_qgl', 'jet2_nTrigMuons']].tolist(), 
             (1 if true_pair == [0, 1] else 0)),  # True pair check
            # Jet 1 and Jet 3
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_qgl', 'jet1_nTrigMuons',
                  'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_qgl', 'jet3_nTrigMuons']].tolist(),
             (1 if true_pair == [0, 2] else 0)),
            # Jet 1 and Jet 4
            (row[['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_qgl', 'jet1_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_qgl', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [0, 3] else 0)),
            # Jet 2 and Jet 3
            (row[['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_qgl', 'jet2_nTrigMuons',
                  'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_qgl', 'jet3_nTrigMuons']].tolist(),
             (1 if true_pair == [1, 2] else 0)),
            # Jet 2 and Jet 4
            (row[['jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_qgl', 'jet2_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_qgl', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [1, 3] else 0)),
            # Jet 3 and Jet 4
            (row[['jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_btagDeepFlavB', 'jet3_qgl', 'jet3_nTrigMuons',
                  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_mass', 'jet4_btagDeepFlavB', 'jet4_qgl', 'jet4_nTrigMuons']].tolist(),
             (1 if true_pair == [2, 3] else 0))
        ]
        
        # Append each pair's features and its corresponding label
        for features, label in jet_pairs:
            pairs.append(features + [label])  # Add the binary label

    return pairs

# Apply the function to your dataframe
jet_pairs = create_jet_pairs(df)

# Create a new dataframe with jet pairs
columns = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_btagDeepFlavB', 'jet1_qgl', 'jet1_nTrigMuons',
           'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_btagDeepFlavB', 'jet2_qgl', 'jet2_nTrigMuons',
           'is_true_pair']  # Add a column for the binary label

pairs_df = pd.DataFrame(jet_pairs, columns=columns)

# View the new dataframe
print(pairs_df.head())


# %%

from sklearn.model_selection import train_test_split
import xgboost as xgb

# Assuming pairs_df is the DataFrame you created
X = pairs_df.drop(columns=['is_true_pair'])  # Features
y = pairs_df['is_true_pair']  # Labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 5,
    'seed': 42
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# %%
# Predict on the test set
y_pred_prob = bst.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]  # Convert probabilities to binary labels

# You can also check accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# %%
