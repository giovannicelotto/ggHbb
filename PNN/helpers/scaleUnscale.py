from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np


def scale(data, featuresForTraining, scalerName, fit=False, weights=None, boosted=False):
    """
    Apply log transformation to features containing 'pt', 'mass', or named 'ht'.
    
    Parameters:
    - data (pd.DataFrame): The dataset.
    - featuresForTraining (list of str): List of feature names to be scaled.
    - scalerName (str): Path to save/load the scaler.
    - fit (bool): If True, fit the scaler; otherwise, load and apply it.
    - weights (array-like, optional): Weights for mean computation (if used in fitting).
    
    Returns:
    - pd.DataFrame: The transformed and scaled dataset.
    """
    
    data = data.astype(np.float32).copy()

    # Apply log transformation to selected features
    log_features = [col for col in featuresForTraining 
                    if ("_pt" in col or "_mass" in col or col == "ht") 
                    and "normalized" not in col]
    #Dijet Mass not used as feature
    #if boosted:
    #    print("Removing Dijet Mass from features")
    #    log_features.remove('dijet_mass')

    for col in log_features:
        print(f"Feature: {col} | Min: {data[col].min():.1f}, Max: {data[col].max():.1f}")
        data.loc[:, col] = np.log1p(data[col])  # np.log1p(x) is equivalent to np.log(1+x)

    # Select features for scaling (excluding 'sf')
    scale_features = [col for col in featuresForTraining if col != 'sf']
    
    if fit:
        # Initialize and fit the scaler
        scaler = preprocessing.StandardScaler()
        if weights is not None:
            scaler.fit(data[scale_features], sample_weight=weights)
        else:
            scaler.fit(data[scale_features])

        # Save the fitted scaler
        with open(scalerName, 'wb') as file:
            pickle.dump({'type': 'standard', 'scaler': scaler}, file)
    
    else:
        # Load the pre-trained scaler
        with open(scalerName, 'rb') as file:
            scaler = pickle.load(file)['scaler']

    # Merge scaled features back into the original DataFrame
    scaled_array = scaler.transform(data[scale_features])
    scaled_data = pd.DataFrame(scaled_array, columns=scale_features, index=data.index, dtype=np.float32)

    # Merge scaled features back into the original DataFrame
    data.update(scaled_data)

    return data
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