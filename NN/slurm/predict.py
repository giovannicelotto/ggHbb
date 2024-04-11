# predict.py
import sys, re
import pandas as pd
from tensorflow.keras.models import load_model
sys.path.append("/t3home/gcelotto/ggHbb/NN")
from NN_multiclass import getFeatures
from helpersForNN import preprocessMultiClass, scale, unscale

def predict(file_path, isMC):
    featuresForTraining, columnsToRead = getFeatures()
    model = load_model("/t3home/gcelotto/ggHbb/NN/output/multiClass/model/model.h5")

    # Load data from file_path and preprocess it as needed
    Xtest = pd.read_parquet(file_path, columns=columnsToRead)
    data = [Xtest]
    data = preprocessMultiClass(data)

    # Perform prediction
    # scale

    print(data[0].jet1_pt.min())
    data[0]  = scale(data[0], scalerName= "/t3home/gcelotto/ggHbb/NN/input/multiclass/myScaler.pkl" ,fit=False)

    print(data[0].columns)
    predictions = model.predict(data[0][featuresForTraining])

    return predictions

if __name__ == "__main__":
    file_path   = sys.argv[1]
    isMC        = sys.argv[2]
    predictions = predict(file_path, isMC)
    predictions = pd.DataFrame(predictions, columns=['DataS', 'ZS', 'ggHS'])
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
    predictions.to_parquet("/scratch/y%d_%d.parquet"%(int(isMC), number))  
    print("saved in /scratch/y%d_%d.parquet"%(int(isMC), number))
