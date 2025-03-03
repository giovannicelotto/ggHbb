from functions import loadMultiParquet, getCommonFilters
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from preprocessMultiClass import preprocessMultiClass
from functions import cut
def match_relative_distribution(df_target, df_source, column='feature', bins = np.linspace(45, 300, 51)):
    """
    Sample rows from df_source to match the relative frequency distribution of df_target
    in the specified column.

    Parameters:
    - df_target (pd.DataFrame): Reference DataFrame with desired distribution.
    - df_source (pd.DataFrame): Source DataFrame to be sampled.
    - column (str): Column to match distribution.
    - num_bins (int): Number of bins to divide the column into.

    Returns:
    - pd.DataFrame: A subset of df_source with matched distribution.
    """
    # Create bins for both dataframes
    df_target['bins'] = pd.cut(df_target[column], bins=bins)
    df_source['bins'] = pd.cut(df_source[column], bins=bins)

    
    # Compute relative bin frequencies
    target_distribution = df_target['bins'].value_counts(normalize=True)
    source_distribution = df_source['bins'].value_counts(normalize=True)

    # Compute the minimum bin ratio (smallest ratio of available-to-needed samples)
    ratio = (source_distribution / target_distribution).dropna()  # Avoid NaN divisions
    min_bin_ratio = ratio.min()  # Find the most difficult-to-populate bin

    # Compute absolute sample sizes based on the minimum bin's ratio
    total_samples = int(min_bin_ratio * len(df_source))  # Scale total sample count
    sample_counts = (target_distribution * total_samples).astype(int)

    print("Target distribution:\n", target_distribution)
    print("Source distribution:\n", source_distribution)
    print("Sampling ratios per bin:\n", ratio)
    print("Minimum bin ratio:", min_bin_ratio)
    print("Final sample counts per bin:\n", sample_counts)
    
    # Compute the number of samples to draw from each bin in df_source
    #total_samples = len(df_source)
    #sample_counts = (target_distribution * total_samples).astype(int)
    #print("SAmple counts")
    #print(sample_counts)
    #print("Items ", sample_counts.items())
    
    # Sample from df_source according to the computed sample counts
    sampled_dfs = []
    for bin_label, count in sample_counts.items():
        bin_df = df_source[df_source['bins'] == bin_label]
        if len(bin_df) >= count:
            sampled_dfs.append(bin_df.sample(n=count, replace=False))
        else:
            sampled_dfs.append(bin_df)  # If not enough samples, take all available
    
    # Concatenate sampled bins and drop the temporary bin column
    df_sampled = pd.concat(sampled_dfs).drop(columns=['bins'], axis=1)

    #print("df_source bin distribution:\n", df_source['bins'].value_counts())

    #df_sampled = df_sampled.drop(['bins'], axis=1)
    return df_sampled

def loadData_adversarial(nReal, nMC, size, outFolder, columnsToRead, featuresForTraining,
                         test_split, advFeature='jet1_btagDeepFlavB', drop=True, boosted=False):

    nData, nHiggs = int(size), int(14e3)

    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    if boosted:
        paths = [
            flatPathCommon + "/Data1A/others",
            flatPathCommon + "/GluGluHToBB/others"]
    else:
        paths = [
            flatPathCommon + "/Data1A/training",
            flatPathCommon + "/GluGluHToBB/others"]
    massHypothesis = [50, 70, 100, 200, 300]
    for m in massHypothesis:
        paths.append(flatPathCommon + "/GluGluH_M%d_ToBB"%(m))
    
    

    dfs = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=False)
    if boosted:
        dfs = cut(dfs, 'dijet_pt', 100, None)
    else:
        dfs = cut(dfs, 'dijet_pt', None, 100)

    # Uncomment in case you want discrete parameter for mass hypotheses
    # 50bb
    dfs[2] = dfs[2][dfs[2].dijet_mass<120]
    # 70bb
    dfs[3] = dfs[3][dfs[3].dijet_mass<140]
    dfs = preprocessMultiClass(dfs)
    
    # method v1
    #dfs[0]['massHypo'] = dfs[0]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])
    #for idx, df in enumerate(dfs[1:]):
    #    dfs[idx+1]['massHypo'] = massHypothesis[idx]
    #    print("Process %d Mass %d"%(idx, massHypothesis[idx]))

    if 'massHypo' in featuresForTraining:
        massHypothesis = np.array([125]+massHypothesis)
        for idx, df in enumerate(dfs):
            dfs[idx]['massHypo'] = dfs[idx]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])


    
    
    # Each sample has the same number of elements (note in principle you could use 1/5 of Higgs data since you have 5 samples)
    for idx, df in enumerate(dfs):
        if idx==0:
            dfs[idx] = df.head(nData)
        else:
            dfs[idx] = df.head(nHiggs)
    genMass = np.concatenate([np.zeros(len(dfs[0])),
               np.ones(len(dfs[1]))*massHypothesis[0],
               np.ones(len(dfs[2]))*massHypothesis[1],
               np.ones(len(dfs[3]))*massHypothesis[2],
               np.ones(len(dfs[4]))*massHypothesis[3],
               np.ones(len(dfs[5]))*massHypothesis[4],
               np.ones(len(dfs[6]))*massHypothesis[5]])
    for idx, df in enumerate(dfs):
        print("%d elements in df %d"%(len(df), idx))
    # Create the labels for Background (0) and Signal (1)
    lenBkg  = len(dfs[0])
    lenS = 0
    for df in dfs[1:]:
        lenS = lenS + len(df)
    Y_0 = pd.DataFrame(np.zeros(lenBkg))
    Y_1 = pd.DataFrame(np.ones(lenS))

    # Each sample has sum = 1. In case it is not possible to have the same number of events
    Ws = [np.ones(lenBkg)]
    for df in dfs[1:]:
        Ws.append(df.sf)    
    for idx,W in enumerate(Ws):
        Ws[idx] = Ws[idx]/np.sum(Ws[idx])
        print(Ws[idx].sum())

    # For Signal create one unique array of weights in order to have a total weight of 1 for Signal and not nSamples*1
    W_H = np.concatenate(Ws[1:])
    W_H = W_H/np.sum(W_H)

    
    Y = np.concatenate((Y_0, Y_1))
    X = pd.concat(dfs)
    W = np.concatenate([Ws[0], W_H])
    
    X, Y, W, genMass = shuffle(X, Y, W, genMass, random_state=1999)

    Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest = train_test_split(X, Y, X[advFeature], W, genMass, test_size=test_split, random_state=1999)
    if drop:
        Xtrain = Xtrain.drop([advFeature], axis=1)
        Xtest = Xtest.drop([advFeature], axis=1)

    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)

    Ytrain, Ytest, Wtrain, Wtest = Ytrain.reshape(-1), Ytest.reshape(-1), Wtrain.reshape(-1), Wtest.reshape(-1)
    return Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest


# New function with sampling

def uniform_sample(df, column='dijet_mass', num_bins=20):
    """
    Sample rows from a DataFrame to ensure a uniform distribution in the specified column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column to sample uniformly.
    - num_bins (int): Number of bins to divide the column into.

    Returns:
    - pd.DataFrame: A subset of df with uniform distribution in the specified column.
    """

    # Create bins based on the target column
    df['bins'] = pd.cut(df[column], bins=num_bins)

    # Find the smallest bin size to ensure fair sampling
    min_bin_size = df['bins'].value_counts().min()

    # Sample an equal number of rows from each bin
    df_uniform = df.groupby('bins', group_keys=False).apply(lambda x: x.sample(n=min_bin_size, replace=False))

    # Drop the temporary bin column
    df_uniform = df_uniform.drop(columns=['bins'], axis=1)

    return df_uniform
def loadData_sampling(nReal, nMC, size, outFolder, columnsToRead, featuresForTraining,
                         test_split, advFeature='jet1_btagDeepFlavB', drop=True, boosted=False):

    nData, nHiggs = int(size), int(20e3)

    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    
    if boosted:
        paths = [
            flatPathCommon + "/Data1A/others",
            flatPathCommon + "/GluGluHToBB/others"]
    else:
        paths = [
            flatPathCommon + "/Data1A/training",
            flatPathCommon + "/GluGluHToBB/training"]
    
    massHypothesis = [50, 70, 100, 200, 300]
    for m in massHypothesis:
        paths.append(flatPathCommon + "/GluGluH_M%d_ToBB"%(m))
    
    
    
    btagTight=True if boosted else False
    dfs = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=False, filters=getCommonFilters(btagTight=btagTight))
    if boosted==1:
        dfs = cut(dfs, 'dijet_pt', 100, 160)
        dfs = cut(dfs, 'dijet_mass', 50, None)
    elif boosted==2:
        dfs = cut(dfs, 'dijet_pt', 160, None)
        dfs = cut(dfs, 'dijet_mass', 50, None)
    else:
        dfs = cut(dfs, 'dijet_pt', None, 100)
        dfs = cut(dfs, 'dijet_mass', 45, None)

    # Uncomment in case you want discrete parameter for mass hypotheses
    # 50bb
    dfs[2] = dfs[2][dfs[2].dijet_mass<120]
    # 70bb
    dfs[3] = dfs[3][dfs[3].dijet_mass<140]
    #if boosted:
    #    # 300 GeV
    #    dfs[-1] = dfs[-1][dfs[-1].dijet_mass>140]
    #    # 100 GeV
    #    dfs[4] = dfs[4][dfs[4].dijet_mass<150]
    dfs = preprocessMultiClass(dfs)
    # Removing Higgs at 300 GEV
    #dfs[-1] = dfs[-1].iloc[:nHiggs]
    #dfs[4] = dfs[4].iloc[:int(nHiggs*1.5)]

    massHypothesis = np.array([125]+massHypothesis)
    if 'massHypo' in featuresForTraining:
        for idx, df in enumerate(dfs):
            dfs[idx]['massHypo'] = dfs[idx]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])


    

    genMass = np.concatenate([np.zeros(len(dfs[0])),
               np.ones(len(dfs[1]))*massHypothesis[0],
               np.ones(len(dfs[2]))*massHypothesis[1],
               np.ones(len(dfs[3]))*massHypothesis[2],
               np.ones(len(dfs[4]))*massHypothesis[3],
               np.ones(len(dfs[5]))*massHypothesis[4],
               np.ones(len(dfs[6]))*massHypothesis[5]])
    
    dfSig = pd.concat(dfs[1:])
    dfBkg = dfs[0]

    dfSig['genMass'] = np.concatenate([
               np.ones(len(dfs[1]))*massHypothesis[0],
               np.ones(len(dfs[2]))*massHypothesis[1],
               np.ones(len(dfs[3]))*massHypothesis[2],
               np.ones(len(dfs[4]))*massHypothesis[3],
               np.ones(len(dfs[5]))*massHypothesis[4],
               np.ones(len(dfs[6]))*massHypothesis[5]])
    dfBkg['genMass'] = np.zeros(len(dfs[0]))

    dfBkg['Y']=0
    dfSig['Y']=1

    if boosted:
        dfSig_sampled = uniform_sample(dfSig, column='dijet_mass', num_bins=101)
        dfBkg_sampled = uniform_sample(dfBkg, column='dijet_mass', num_bins=101)
    else:
        dfBkg_sampled = dfBkg.copy()
        dfSig_sampled = match_relative_distribution(dfBkg, dfSig, column='dijet_mass', bins = np.linspace(45, 300, 51))
        del dfBkg

    for m in massHypothesis:
        print("%d elements in df %d"%(len(dfSig_sampled[dfSig_sampled.genMass==m]), m))
    print("%d elements in df %d"%(len(dfBkg_sampled[dfBkg_sampled.genMass==0]), 0))

    dfSig_sampled['W']=dfSig_sampled.sf/dfSig_sampled.sf.sum()
    dfBkg_sampled['W']=np.ones(len(dfBkg_sampled))/len(dfBkg_sampled)

    X = pd.concat([dfSig_sampled, dfBkg_sampled])
    Y = np.array(X['Y'].values)
    W = np.array(X.W.values)
    genMass = np.array(X.genMass.values)

    X = X.drop(['Y', 'W', 'genMass'], axis=1)
    
    X, Y, W, genMass = shuffle(X, Y, W, genMass, random_state=1999)

    Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest = train_test_split(X, Y, X[advFeature], W, genMass, test_size=test_split, random_state=1999)
    if drop:
        Xtrain = Xtrain.drop([advFeature], axis=1)
        Xtest = Xtest.drop([advFeature], axis=1)

    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)

    Ytrain, Ytest, Wtrain, Wtest = Ytrain.reshape(-1), Ytest.reshape(-1), Wtrain.reshape(-1), Wtest.reshape(-1)
    return Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest