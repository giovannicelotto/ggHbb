# %%
import xgboost as xgb
import pandas as pd
from helpers.train_and_auc import train_and_auc
from helpers.loadData_pairs import loadData_pairs
from bayes_opt import BayesianOptimization
# %%
# open the dataframes and keep the same amount for mass hypothesis
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs"

X_train, X_test, y_train, y_test = loadData_pairs(path=path)

# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

pbounds = {
        'depth': (3, 10),
        'eta': (0.01, 0.3),
        'num_boost_round': (100,1000)}
optimizer = BayesianOptimization(
f=lambda depth, eta, num_boost_round:train_and_auc(dtrain=dtrain, dtest=dtest, y_test=y_test, depth=depth, eta=eta, num_boost_round=num_boost_round),
pbounds=pbounds,
verbose=2, 
random_state=1,
allow_duplicate_points=True
)
    
optimizer.maximize(
    init_points=3,
    n_iter=5,
)