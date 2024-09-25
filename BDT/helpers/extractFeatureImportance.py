import xgboost as xgb
bst = xgb.Booster()
bst.load_model('/t3home/gcelotto/ggHbb/BDT/model/xgboost_jet_model_optimal.model')

def getImportance(bst):
    importance = bst.get_score(importance_type='weight')

    # Print the feature importance
    for feature, score in importance.items():
        print(f'Feature: {feature}, Score: {score}')

getImportance(bst=bst)