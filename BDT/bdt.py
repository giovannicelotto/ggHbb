# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
import glob
import pandas as pd
# %%
# open the dataframes and keep the same amount for mass hypothesis
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs"
def load_dfs(path):
    dfs = [ ]
    fileNames = glob.glob(path+"/*.parquet")
    min_len = 0
    for fileName in fileNames:
        df = pd.read_parquet(fileName)
        min_len = len(df) if ((len(df)<min_len) | (min_len==0)) else min_len
        dfs.append(df)
    for idx, df in enumerate(dfs):
        dfs[idx] = dfs[idx].head(min_len)
    return dfs, min_len

dfs, min_len = load_dfs(path)
# concatenate the dataframes
df = pd.concat(dfs)
print("df head")
print(df.head())
# %%

# Assuming pairs_df is the DataFrame you created
X = df.drop(columns=['is_true_pair'])  # Features
y = df['is_true_pair']  # Labels

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)
print("Xtrain")
print(X_train.head())
# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
#
## Set the parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 5,
    'seed': 42
}
#
## Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)
bst.save_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model.model')
#
## %%
## Predict on the test set
bst_loaded = xgb.Booster()
bst_loaded.load_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model.model')
y_pred_prob = bst_loaded.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]  # Convert probabilities to binary labels
#
## You can also check accuracy
from sklearn.metrics import accuracy_score
#
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
#
## %%
#