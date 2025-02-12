from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np


def scale(data, featuresForTraining, scalerName, fit=False, weights=None):
    '''
    apply log to features with pt and mass in name
      
    '''
    for colName in featuresForTraining:
        if (("_pt" in colName) | ("_mass" in colName) | (colName=="ht")) :
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


def test_gaussianity_validation(Xtrain, Xval, featuresForTraining, inFolder):

    import matplotlib.pyplot as plt
    train_means = Xtrain[featuresForTraining].mean(axis=0)
    train_stds = Xtrain[featuresForTraining].std(axis=0)
    val_means = Xval[featuresForTraining].mean(axis=0)
    val_stds = Xval[featuresForTraining].std(axis=0)

    # Create a plot for means with error bars
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Plotting the training set means and standard deviations
    ax.errorbar(range(len(featuresForTraining)), train_means, yerr=train_stds/np.sqrt(len(Xval)), fmt='o', label='Train Set', color='blue', capsize=5)
    ax.errorbar(np.arange(len(featuresForTraining))+0.2, val_means, yerr=val_stds/np.sqrt(len(Xval)), fmt='o', label='Validation Set', color='red', capsize=5)
    from scipy.stats import shapiro

    stat_val, p_val = shapiro(val_means)
    print("Shapiro-Wilk Test for Validation Set Means: Stat =", stat_val, "p-value =", p_val)

    if p_val > 0.05:
        print("Validation Set Means follow a Gaussian distribution (fail to reject H0).")
    else:
        print("Validation Set Means do not follow a Gaussian distribution (reject H0).")
    ax.text(x=0.05, y=0.1, s="Shapiro-Wilk p-value : %.2f"%p_val, transform=ax.transAxes)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(range(len(featuresForTraining)), featuresForTraining, rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Values')
    ax.legend()
    fig.savefig(inFolder+"/test_gaussianity_validation_set.png", bbox_inches='tight')