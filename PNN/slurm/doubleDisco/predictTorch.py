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

# change the model
# change the features
# change the scaler

def predict(file_path, modelName):
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s/model"%(modelName)
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s/featuresForTraining.npy"%modelName)
    mass_bins = np.load("/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s/mass_bins.npy"%modelName)
    mass_bins[0]=40
    print(mass_bins)
    nn1 = torch.load(modelDir+"/nn1.pth", map_location=torch.device('cpu'))
    nn2 = torch.load(modelDir+"/nn2.pth", map_location=torch.device('cpu'))

    # Load data from file_path and preprocess it as needed
    print(file_path)
    Xtest = pd.read_parquet(file_path,
                                engine='pyarrow',
                                 filters= getCommonFilters()        )
    mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    
    data = [Xtest]
    data = cut(data, 'dijet_pt', None, 100)

    data = preprocessMultiClass(data)
    Xtest = data[0]
    Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])

    if 'bin_center' in featuresForTraining:
        bin_centers = [(mass_bins[i] + mass_bins[i+1]) / 2 for i in range(len(mass_bins) - 1)]


        bin_indices = np.digitize(Xtest['dijet_mass'].values, mass_bins) - 1
        Xtest['bin_center'] = np.where(
            (bin_indices >= 0) & (bin_indices < len(bin_centers)),
            np.array(bin_centers)[bin_indices],
            np.nan  # Assign NaN for out-of-range dijet_mass
        )

    # Perform prediction
    # scale
    Xtest  = scale(Xtest, featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
    nn1.eval()
    nn2.eval()
    data_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()
    

    with torch.no_grad():  # No need to track gradients for inference
        data_predictions1 = nn1(data_tensor).numpy()
        data_predictions2 = nn2(data_tensor).numpy()
        
    

    return data_predictions1, data_predictions2

if __name__ == "__main__":
    file_path   = sys.argv[1]
    process        = sys.argv[2]
    modelName        = sys.argv[3]

    
    predictions1, predictions2 = predict(file_path, modelName)
    print("Shape of predictions1", predictions1.shape)
    predictions = pd.DataFrame({
    'PNN1': predictions1.reshape(-1),
    'PNN2': predictions2.reshape(-1)
})
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/yDD_%s_FN%d.parquet"%(process, number))
    print("saved in /scratch/yDD_%s_FN%d.parquet"%(process, number))
