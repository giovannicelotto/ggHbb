from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np


def scale(data, scalerName, fit=False, weights=None):
    
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            print("feature: %s min: %.1f max: %.1f"%(colName, data[colName].min(), data[colName].max()))
            data[colName] = np.log(1+data[colName])
            #print("log done for %s"%colName)
# fitting the scaler
    if fit:
        scaler  = preprocessing.StandardScaler().fit(data[[col for col in data.columns if col!='sf']])
        if weights is not None:
            scaled_array = scaler.transform(data[[col for col in data.columns if col!='sf']], sample_weights=weights)
        else:
            scaled_array = scaler.transform(data[[col for col in data.columns if col!='sf']])
        scalers = {
            'type'  : 'standard',
            'scaler': scaler,
            }
        with open(scalerName, 'wb') as file:
            pickle.dump(scalers, file)
# Scaling without fitting
    else:
        with open(scalerName, 'rb') as file:
            scalers = pickle.load(file)
            scaler = scalers['scaler']
            if weights is not None:
                scaled_array = scaler.transform(data[[col for col in data.columns if col != 'sf']], sample_weight=weights)
            else:
                scaled_array = scaler.transform(data[[col for col in data.columns if col != 'sf']])
            
    dataScaled = pd.DataFrame(scaled_array, columns=[col for col in data.columns if col!='sf'], index=data.index)
    dataScaled['sf'] = data['sf']
    
    return dataScaled

def unscale(data, scalerName):
    with open(scalerName, 'rb') as file:
        scalers = pickle.load(file)
        scaler = scalers['scaler']
        scaled_array = scaler.inverse_transform(data[[col for col in data.columns if col!='sf']])
        dataUnscaled = pd.DataFrame(scaled_array, columns=[col for col in data.columns if col!='sf'], index=data.index)
        dataUnscaled['sf'] = data['sf']

    
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            dataUnscaled[colName] = np.exp(dataUnscaled[colName]) - 1
    
    return dataUnscaled
