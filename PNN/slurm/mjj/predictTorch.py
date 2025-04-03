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

def predict(file_path, modelName, boosted):
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model"%(modelName)
    featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/featuresForTraining.npy"%modelName))
    print("Before opening model", flush=True)
    nn = torch.load(modelDir+"/model.pth", map_location=torch.device('cpu'))

    # Load data from file_path and preprocess it as needed
    print(file_path)
    print("Before opening", flush=True)
    Xtest = pd.read_parquet(file_path,
                                engine='pyarrow',
                                 filters= getCommonFilters()        )
    print("File opened", flush=True)
    #mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    #Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])
    #featuresForTraining = featuresForTraining + ['dijet_mass']
    
    data = [Xtest]
    if int(boosted)==1:
        data = cut(data, 'dijet_pt', 100, 160)
    elif int(boosted)==2:
        data = cut(data, 'dijet_pt', 160, None)
    elif int(boosted)==60:
        data = cut(data, 'dijet_pt', 60, 100)
    print("File cut")
    data = preprocessMultiClass(data)

    # Perform prediction
    # scale
    print("Scaling")
    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
    nn.eval()
    data_tensor = torch.tensor(np.float32(data[0][featuresForTraining].values)).float()
    

    with torch.no_grad():  # No need to track gradients for inference
        data_predictions1 = nn(data_tensor).numpy()
        
    

    return data_predictions1

if __name__ == "__main__":
    file_path   = sys.argv[1]
    process        = sys.argv[2]
    modelName        = sys.argv[3]
    boosted        = int(sys.argv[4])

    print("Arguments passed")
    predictions1 = predict(file_path, modelName, boosted)
    print("Shape of predictions1", predictions1.shape)
    predictions = pd.DataFrame({
    'PNN': predictions1.reshape(-1),
})
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
        print("number matched ", number)
    predictions.to_parquet("/scratch/yMjj_%s_FN%d.parquet"%(process, number))
    print("saved in /scratch/yMjj%s_FN%d.parquet"%(process, number))
