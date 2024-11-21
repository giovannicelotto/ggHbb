# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from helpers.load_dfPairs_sameLen import load_dfPairs_sameLen
from helpers.bdt_train_save import bdt_train_save
from helpers.loadData_pairs import loadData_pairs
# %%
# open the dataframes and keep the same amount for mass hypothesis
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/gen4JetsFeatures/df_pairs"
X_train, X_val, X_test, y_train, y_val, y_test = loadData_pairs(path=path)


# %%
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

#bdt_train_save(dtrain=dtrain, dtest=dtest, y_test=y_test, depth = 7, eta = 0.079, num_boost_round=572, outName='/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')
# %%
depth = 7
eta = 0.079
num_boost_round=572
outName='/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model'
depth = int(depth)
num_boost_round = int(num_boost_round)
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss','error'],
    'eta': eta,
    'max_depth': depth,
    'seed': 42,
}
# %%
## Train the model
evals_result = {} # dictionary to save the loss
watchlist = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    evals_result=evals_result,
    early_stopping_rounds=50  # Stop if no improvement in 10 rounds
)
bst.save_model(outName)


## Predict on the test set
bst_loaded = xgb.Booster()
bst_loaded.load_model(outName)
y_pred_prob = bst.predict(dtest)

## You can also check accuracy

#accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
print(depth, eta, num_boost_round)
print("AUC", auc)
# %%
import matplotlib.pyplot as plt
import numpy as np
# Extract the loss history from evals_result
train_loss = evals_result['train']['logloss']
validation_loss = evals_result['eval']['logloss']
train_accuracy = evals_result['train']['error']
validation_accuracy = evals_result['eval']['error']

# Plot the loss history
fig, ax = plt.subplots(1, 1)
ax.plot(train_loss, label='Training Loss', color='blue')
ax.plot(validation_loss, label='Validation Loss', color='orange')

ax.plot(np.array(train_accuracy), label='Training Error', color='blue')
ax.plot(np.array(validation_accuracy), label='Validation Error', color='orange')

ax.set_title('Training and Validation Loss Over Epochs', fontsize=14)
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Log Loss', fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 0.2)

ax.grid(True)
# %%
