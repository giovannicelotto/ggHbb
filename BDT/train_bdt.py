# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from helpers.load_dfPairs_sameLen import load_dfPairs_sameLen
from helpers.load_dfPairs_sameLen import bdt_train_save
# %%
# open the dataframes and keep the same amount for mass hypothesis
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs"


dfs, min_len = load_dfPairs_sameLen(path)

# concatenate the dataframes
df = pd.concat(dfs)

# %%

# Assuming pairs_df is the DataFrame you created
X = df.drop(columns=['is_true_pair', 'massHypo'])  # Features
y = df['is_true_pair']  # Labels

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)

# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
bdt_train_save(dtrain=dtrain, dtest=dtest, y_test=y_test, depth = 7, eta = 0.079, num_boost_round=572, outName='/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')
# %%
## Set the parameters for XGBoost
def eval_performance(depth, eta, num_boost_round):
    depth = int(depth)
    num_boost_round = int(num_boost_round)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': eta,
        'max_depth': depth,
        'seed': 42,
    }
    #
    ## Train the model
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    #bst.save_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')
    #

    ## Predict on the test set
    #bst_loaded = xgb.Booster()
    #bst_loaded.load_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')
    y_pred_prob = bst.predict(dtest)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]  # Convert probabilities to binary labels
    #
    ## You can also check accuracy
    #
    #accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    print(depth, eta, num_boost_round)
    print(auc)
    #print(f"Accuracy: {accuracy:.4f}")
    return auc
#
## %%



# %%




# %%
#from bayes_opt import BayesianOptimization
#pbounds = {
#        'depth': (3, 10),
#        'eta': (0.01, 0.3),
#        'num_boost_round': (100,1000)}
#optimizer = BayesianOptimization(
#f=eval_performance,
#pbounds=pbounds,
#verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#random_state=1,
#allow_duplicate_points=True
#)
#    
#optimizer.maximize(
#    init_points=40,
#    n_iter=50,
#)