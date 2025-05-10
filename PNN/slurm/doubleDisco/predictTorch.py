# predict.py
import sys, re
import pandas as pd
import torch
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.scaleUnscale import scale
from helpers.getFeatures import getFeatures
import numpy as np
from helpers.dcorLoss import Classifier
from functions import getCommonFilters, cut
def get_layer_sizes(state_dict, n_input_features):
        layer_sizes = []
        current_features = n_input_features

        for key, tensor in state_dict.items():
            if "weight" in key and tensor.dim() == 2:
                weight_shape = state_dict[key].shape
                if weight_shape[1] == current_features:  # Ensure it's a valid layer
                    layer_sizes.append(weight_shape[0])
                    current_features = weight_shape[0]  # Update for the next layer

        return layer_sizes


def predict(file_path, modelName, multigpu, epoch):
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    # Load data from file_path and preprocess it as needed
    print(file_path)
    Xtest = pd.read_parquet(file_path,
                                engine='pyarrow',
                                 filters= getCommonFilters()        )
    outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%(modelName)
    featuresForTraining = np.load(outFolder+"/featuresForTraining.npy")
    mass_bins = np.load(outFolder+"/mass_bins.npy") 

    print(mass_bins)
    if multigpu==0:
        nn1 = torch.load(outFolder+"/model/nn1.pth", map_location=torch.device('cpu'))
        nn2 = torch.load(outFolder+"/model/nn2.pth", map_location=torch.device('cpu'))
        nn1.eval()
        nn2.eval()
    else:
        state_dict1 = torch.load(outFolder + "/model/nn1_e%d.pth"%epoch, map_location=torch.device('cpu'))
        state_dict2 = torch.load(outFolder + "/model/nn2_e%d.pth"%epoch, map_location=torch.device('cpu'))
        ## Remove the 'module.' prefix if it exists
        state_dict1 = {k.replace('module.', ''): v for k, v in state_dict1.items()}
        state_dict2 = {k.replace('module.', ''): v for k, v in state_dict2.items()}
        layer_sizes1 = get_layer_sizes(state_dict1, n_input_features=len(featuresForTraining))
        layer_sizes2 = get_layer_sizes(state_dict2, n_input_features=len(featuresForTraining))
        print("nNodes", layer_sizes1[:-1])
        nn1 = Classifier(input_dim=len(featuresForTraining), nNodes=layer_sizes1[:-1])
        nn2 = Classifier(input_dim=len(featuresForTraining), nNodes=layer_sizes2[:-1])

        ## Now load the state_dict into the model
        nn1.load_state_dict(state_dict1)
        nn2.load_state_dict(state_dict2)

        nn1.eval()
        nn2.eval()

    
    mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    
    data = [Xtest]
    data = cut(data, 'dijet_pt', 100, 160)

    data = preprocessMultiClass(data)
    Xtest = data[0]
    Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])
    Xtest['jet1_btagTight'] = Xtest['jet1_btagDeepFlavB']>0.71
    Xtest['jet2_btagTight'] = Xtest['jet2_btagDeepFlavB']>0.71

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
    Xtest  = scale(Xtest, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
    nn1.eval()
    nn2.eval()
    data_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()
    

    with torch.no_grad():  # No need to track gradients for inference
        data_predictions1 = nn1(data_tensor).numpy()
        data_predictions2 = nn2(data_tensor).numpy()
        
    

    return data_predictions1, data_predictions2

if __name__ == "__main__":
    file_path   = sys.argv[1]
    process     = sys.argv[2]
    modelName   = sys.argv[3]
    multigpu    = int(sys.argv[4])
    epoch    = int(sys.argv[5])
    print(file_path, process, modelName, multigpu, epoch)

    
    predictions1, predictions2 = predict(file_path, modelName, multigpu, epoch)
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
