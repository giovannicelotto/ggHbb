# predict.py
import sys, re
import pandas as pd
import torch
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.scaleUnscale import scale
from helpers.getFeatures import getFeatures
import numpy as np
from functions import getCommonFilters, cut
import joblib
import yaml

# change the model
# change the features
# change the scaler

def predict(file_path, modelName, boosted, quantile_matching=True, run=2):
    '''
    file_path: path to parquet file (str) or pandas dataframe (pd.DataFrame)
    modelName: name of the model to be used for prediction (str)

    
    '''
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model"%(modelName)
    featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model/featuresForTraining.npy"%modelName))
    print("Before opening model", flush=True)
    nn = torch.load(modelDir+"/model.pth", map_location=torch.device('cpu'))

    # Load data from file_path and preprocess it as needed
    if type(file_path) == str:
        print("Before opening", flush=True)
        Xtest = pd.read_parquet(file_path,
                                    engine='pyarrow',
                                     filters= getCommonFilters(btagWP="L", cutDijet=True, ttbarCR='both'))
    elif isinstance(file_path, pd.DataFrame):
        Xtest = file_path
    print("File opened", flush=True)
    #mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    #Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])
    #featuresForTraining = featuresForTraining + ['dijet_mass']

    data = [Xtest]
    #if run==2:
    #    data = cut(data, 'dijet_mass', 50, 300)
    #data[0]['dimuon_mass'] = np.where(data[0]['dimuon_mass']==-999, 0.106, data[0]['dimuon_mass'])
    #data = preprocessMultiClass(data)

    # Perform prediction
    # scale

    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
    for f in data[0].columns:
        print(f,data[0][f].isna().sum())
    nn.eval()
    data_tensor = torch.tensor(np.float32(data[0][featuresForTraining].values)).float()




    if quantile_matching:
        # Prediction with PCA variation
        #if 'dijet_cs' not in data[0].columns:
        #    data[0]['dijet_cs'] = np.ones(len(data[0]))
            
        #pca = joblib.load("/t3home/gcelotto/ggHbb/tt_CR/PCA/pca_ttbar.pkl")
        #with open("/t3home/gcelotto/ggHbb/tt_CR/PCA/pca_variation.yaml", "r") as f:
        #    variation_config = yaml.safe_load(f)
        #X_new = data[0][featuresForTraining].values
        #Z_new = pca.transform(X_new)
        #Z_new[:, 0] += variation_config["shift"]["pc1"] * variation_config["pc1_sigma"]
        #Z_new[:, 1] += variation_config["shift"]["pc2"] * variation_config["pc2_sigma"]
        #Z_new[:, 2] += variation_config["shift"]["pc3"] * variation_config["pc3_sigma"]

        #X_new_var = pca.inverse_transform(Z_new)
        #df_pca_varied = pd.DataFrame(X_new_var, columns=featuresForTraining, index=data[0].index)
        #pca_varied_tensor = torch.tensor(np.float32(df_pca_varied[featuresForTraining].values)).float()



        # Prediction with Quantile Matching + Copula Space
        

        qt_tt  = joblib.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_tt.pkl")
        qt_ggH = joblib.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_ggH.pkl")

        L_tt   = np.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_tt.npy")
        L_ggH  = np.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_ggH.npy")

        from copula_morph import copula_morph
        X_tt_morphed = pd.DataFrame(copula_morph( data[0][featuresForTraining], qt_tt, qt_ggH, L_tt, L_ggH), columns=featuresForTraining, index=data[0].index)
        quantile_varied_tensor = torch.tensor(np.float32(X_tt_morphed[featuresForTraining].values)).float()

        with torch.no_grad():  # No need to track gradients for inference
            data_predictions1 = nn(data_tensor).numpy()
            #data_predictions_pca_varied = nn(pca_varied_tensor).numpy()
            data_predictions_qm = nn(quantile_varied_tensor).numpy()
        return data_predictions1, data_predictions_qm
    else:
        with torch.no_grad():
            data_predictions1 = nn(data_tensor).numpy()
        return data_predictions1
    
    


if __name__ == "__main__":
    file_path   = sys.argv[1]
    process        = sys.argv[2]
    modelName        = sys.argv[3]
    boosted        = int(sys.argv[4])

    print("Arguments passed")
    quantile_matching=True
    if quantile_matching:
        predictions1, data_predictions_pca_varied, data_predictions_qm = predict(file_path, modelName, boosted,  quantile_matching=quantile_matching)
        print("Shape of predictions1", predictions1.shape)
        predictions = pd.DataFrame({
            'PNN': predictions1.reshape(-1),
            #'PNN_pca': data_predictions_pca_varied.reshape(-1),
            'PNN_qm': data_predictions_qm.reshape(-1)
        })
    else:
        predictions1 = predict(file_path, modelName, boosted, quantile_matching=quantile_matching)
        print("Shape of predictions1", predictions1.shape)
        predictions = pd.DataFrame({
            'PNN': predictions1.reshape(-1),
        #    'PNN_pca': data_predictions_pca_varied.reshape(-1),
        #    'PNN_qm': data_predictions_qm.reshape(-1)
        })
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
        print("number matched ", number)
    name = "/scratch/yMjj_%s_FN%d.parquet"%(process, number)
    predictions.to_parquet(name)
    print(name)
