from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np


def scale(data, featuresForTraining, scalerName, fit=False, weights=None):
    '''
    apply log to features with pt and mass in name
    
    
    
    
    
    '''
    for colName in featuresForTraining:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            if "normalized" in colName:
                #print("normalized found in ", colName)
                continue
            print("feature: %s min: %.1f max: %.1f"%(colName, data[colName].min(), data[colName].max()))
            data[colName] = np.log(1+data[colName])
            #print("log done for %s"%colName)
# fitting the scaler
    if fit:
        if weights is not None:
            scaler  = preprocessing.StandardScaler().fit(data[[col for col in featuresForTraining if col!='sf']], sample_weights=weights)
        else:
            scaler  = preprocessing.StandardScaler().fit(data[[col for col in featuresForTraining if col!='sf']])

        scaled_array = scaler.transform(data[[col for col in featuresForTraining if col!='sf']])
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
            scaled_array = scaler.transform(data[[col for col in featuresForTraining if col != 'sf']])
            
    dataScaled = pd.DataFrame(scaled_array, columns=[col for col in featuresForTraining if col!='sf'], index=data.index)
    for col in data.columns:
        if col not in featuresForTraining:
            dataScaled[col] = data[col]
            
    #dataScaled['sf'] = data['sf']
    
    return dataScaled

def unscale(data, featuresForTraining, scalerName):
    with open(scalerName, 'rb') as file:
        scalers = pickle.load(file)
        scaler = scalers['scaler']
        scaled_array = scaler.inverse_transform(data[[col for col in featuresForTraining if col!='sf']])
        dataUnscaled = pd.DataFrame(scaled_array, columns=[col for col in featuresForTraining if col!='sf'], index=data.index)
        #dataUnscaled['sf'] = data['sf']

    
    for colName in featuresForTraining:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            dataUnscaled[colName] = np.exp(dataUnscaled[colName]) - 1
    
    for feature in data.columns:
        if feature not in featuresForTraining:
            dataUnscaled[feature] = data[feature]
    return dataUnscaled
