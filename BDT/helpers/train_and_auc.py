import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score


def train_and_auc(dtrain, dtest, y_test, depth, eta, num_boost_round):
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

    #

    ## Predict on the test set
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
