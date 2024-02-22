# predict.py
import sys, re
import pandas as pd
from tensorflow.keras.models import load_model

def predict(file_path, isMC):
    featuresForTraining=['jet1_eta', 'jet1_btagDeepFlavB', 'jet1_qgl',
                         'jet2_eta', 'jet2_btagDeepFlavB', 'jet2_qgl',
                         'dijet_eta',
                         'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_twist',
                         'muon_pt',  'nJets',     'ht',   'muon_pfRelIso03_all']
    # Load your trained model
    model = load_model("/t3home/gcelotto/ggHbb/outputs/model_inclusive.h5")

    # Load data from file_path and preprocess it as needed
    Xtest = pd.read_parquet(file_path)
    
    Xtest = Xtest[(Xtest.jet1_pt>20) & (Xtest.jet2_pt>20)]
    Xtest = Xtest[(Xtest.jet1_eta<2.5) & (Xtest.jet1_eta>-2.5)]
    Xtest = Xtest[(Xtest.jet2_eta<2.5) & (Xtest.jet2_eta>-2.5)]

    # Perform prediction
    predictions = model.predict(Xtest[featuresForTraining])

    return predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]
    predictions = predict(file_path, isMC)
    predictions = pd.DataFrame(predictions, columns=['NNoutput'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/t3home/gcelotto/ggHbb/scripts/NN/NNoutputFiles/y%d_%d.parquet"%(int(isMC), number))  
