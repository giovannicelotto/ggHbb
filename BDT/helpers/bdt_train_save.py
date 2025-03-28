import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

def bdt_train_save(dtrain, dtest, y_test, depth, eta, num_boost_round, outName):
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
    bst.save_model(outName)
    

    ## Predict on the test set
    bst_loaded = xgb.Booster()
    bst_loaded.load_model(outName)
    y_pred_prob = bst.predict(dtest)
    
    ## You can also check accuracy

    #accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    print(depth, eta, num_boost_round)
    print(auc)