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

def predict(file_path, isMC, modelName):
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results/%s/model"%(modelName)
    featuresForTraining = np.load(modelDir + "/featuresForTraining.npy")
    model = load_model(modelDir + "/myModel.h5")

    # Load data from file_path and preprocess it as needed
    print(file_path)
    Xtest = pd.read_parquet(file_path,
                                engine='pyarrow',
                                 filters= [
                                    ('jet1_pt', '>',  20),
                                    ('jet2_pt', '>',  20),

                                    ('jet1_mass', '>', 0),
                                    ('jet2_mass', '>', 0),
                                    ('jet3_mass', '>', 0),


                                    ('jet1_eta', '>', -2.5),
                                    ('jet2_eta', '>', -2.5),
                                    ('jet1_eta', '<',  2.5),
                                    ('jet2_eta', '<',  2.5),
                                    ('jet1_btagDeepFlavB', '>',  0.2783),
                                    ('jet2_btagDeepFlavB', '>',  0.2783),
                                          ]        )
    mass_hypo_list = np.array([50, 70, 100, 200, 300, 125])
    
    Xtest['massHypo'] = Xtest['dijet_mass'].apply(lambda x: mass_hypo_list[np.abs(mass_hypo_list - x).argmin()])
    
    featuresForTraining = np.array(list(featuresForTraining) + ['massHypo'])
    data = [Xtest]

    data = preprocessMultiClass(data)

    # Perform prediction
    # scale

    print(data[0].jet1_pt.min())
    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)

    print(data[0].columns)
    predictions = model.predict(data[0][featuresForTraining])

    return predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]
    modelName        = sys.argv[3]

    
    predictions = predict(file_path, isMC, modelName)
    predictions = pd.DataFrame(predictions, columns=['PNNPredictions'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
    print("saved in /scratch/yMC%d_fn%d.parquet"%(int(isMC), number))
