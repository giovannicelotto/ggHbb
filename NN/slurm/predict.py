# predict.py
import sys, re
import pandas as pd
from tensorflow.keras.models import load_model
sys.path.append("/t3home/gcelotto/ggHbb/NN")
from NN_multiclass import getFeatures
from helpersForNN import preprocessMultiClass, scale, unscale

def predict(file_path, isMC, pTClass):
    featuresForTraining, columnsToRead = getFeatures()
    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][pTClass]
    model = load_model("/t3home/gcelotto/ggHbb/NN/output/multiClass/%s/_medium/model/model_medium.h5"%suffix)

    # Load data from file_path and preprocess it as needed
    Xtest = pd.read_parquet(file_path, columns=columnsToRead)
    print(Xtest)
    data = [Xtest]
    print("This is suffix1", suffix)
    data = preprocessMultiClass(data, leptonClass=None, pTmin=pTmin, pTmax=pTmax, suffix=suffix)

    # Perform prediction
    # scale

    print(data[0].jet1_pt.min())
    data[0]  = scale(data[0], scalerName= "/t3home/gcelotto/ggHbb/NN/input/multiclass/%s/myScaler_medium.pkl"%suffix ,fit=False)

    print(data[0].columns)
    predictions = model.predict(data[0][featuresForTraining])

    return predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]
    pTClass     = int(sys.argv[3])
    
    predictions = predict(file_path, isMC, pTClass)
    predictions = pd.DataFrame(predictions, columns=['DataS', 'ZS', 'ggHS'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/yMC%d_fn%d_pt%d.parquet"%(int(isMC), number, pTClass))
    print("saved in /scratch/yMC%d_fn%d_pt%d.parquet"%(int(isMC), number, pTClass))
