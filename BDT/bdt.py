from sklearn.model_selection import train_test_split
import xgboost as xgb
# %%
# open the dataframes and keep the same amount for mass hypothesis

# concatenate the dataframes


# Assuming pairs_df is the DataFrame you created
X = pairs_df.drop(columns=['is_true_pair'])  # Features
y = pairs_df['is_true_pair']  # Labels

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
y_pred_prob = bst.predict(dtest)
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