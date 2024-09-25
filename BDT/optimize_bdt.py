# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from helpers.load_dfPairs_sameLen import load_dfPairs_sameLen
from helpers.bdt_train_save import bdt_train_save
from helpers.train_and_auc import train_and_auc
# %%
# open the dataframes and keep the same amount for mass hypothesis
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs"


dfs, min_len = load_dfPairs_sameLen(path)

# concatenate the dataframes
df = pd.concat(dfs)

# Assuming pairs_df is the DataFrame you created
X = df.drop(columns=['is_true_pair', 'massHypo'])  # Features
y = df['is_true_pair']  # Labels

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)

# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

from bayes_opt import BayesianOptimization
pbounds = {
        'depth': (3, 10),
        'eta': (0.01, 0.3),
        'num_boost_round': (100,1000)}
optimizer = BayesianOptimization(
f=lambda depth, eta, num_boost_round:train_and_auc(dtrain=dtrain, dtest=dtest, y_test=y_test, depth=depth, eta=eta, num_boost_round=num_boost_round),
pbounds=pbounds,
verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
random_state=1,
allow_duplicate_points=True
)
    
optimizer.maximize(
    init_points=3,
    n_iter=5,
)