# predict.py
import sys, re
import pandas as pd
import torch
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.scaleUnscale import scale
from helpers.getFeatures import getFeatures
import numpy as np
from functions import getCommonFilters

# change the model
# change the features
# change the scaler

def predict(file_path, isMC, modelName):
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results/%s/model"%(modelName)
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/PNN/results/Dec16_30/featuresForTraining.npy")
    model = torch.load(modelDir+"/model.pth", map_location=torch.device('cpu'))

    # Load data from file_path and preprocess it as needed
    print(file_path)
    Xtest = pd.read_parquet(file_path,
                                engine='pyarrow',
                                 filters= getCommonFilters()        )
    mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])
    
    data = [Xtest]

    data = preprocessMultiClass(data)

    # Perform prediction
    # scale
    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
    model.eval()
    data_tensor = torch.tensor(np.float32(data[0][featuresForTraining].values)).float()
    

    with torch.no_grad():  # No need to track gradients for inference
        data_predictions = model(data_tensor).numpy()
        
    

    return data_predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]
    modelName        = sys.argv[3]

    
    predictions = predict(file_path, isMC, modelName)
    predictions = pd.DataFrame(predictions, columns=['PNN'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
    print("saved in /scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
