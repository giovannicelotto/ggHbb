# predict.py
import sys, re
import pandas as pd
from tensorflow.keras.models import load_model
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.scaleUnscale import scale
from helpers.getFeatures import getFeatures
import numpy as np

# change the model
# change the features
# change the scaler

def predict(file_path, isMC):
    featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    featuresForTraining = np.load("/t3home/gcelotto/ggHbb/PNN/results/basicFeatures/model/featuresForTraining.npy")
    model = load_model("/t3home/gcelotto/ggHbb/PNN/results/basicFeatures/model/myModel.h5")

    # Load data from file_path and preprocess it as needed
    Xtest = pd.read_parquet(file_path, columns=columnsToRead)
    if int(isMC)==0:
        Xtest['massHypo'] = np.random.choice([50, 70, 100, 200, 300, 125], size=len(Xtest))
    elif int(isMC)==1:
        Xtest['massHypo'] = 125
    featuresForTraining = np.array(list(featuresForTraining) + ['massHypo'])
    data = [Xtest]

    data = preprocessMultiClass(data)

    # Perform prediction
    # scale

    print(data[0].jet1_pt.min())
    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= "/t3home/gcelotto/ggHbb/PNN/results/basicFeatures/model/myScaler.pkl" ,fit=False)

    print(data[0].columns)
    predictions = model.predict(data[0][featuresForTraining])

    return predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]

    
    predictions = predict(file_path, isMC)
    predictions = pd.DataFrame(predictions, columns=['PNNPredictions'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
    print("saved in /scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
