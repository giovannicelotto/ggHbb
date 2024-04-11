
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd
def preprocessMultiClass(dfs):
    '''
    dfs is a list of dataframes
    '''
    
    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    dfs_new = []
    for idx, df in enumerate(dfs):
        df = df[(df.jet1_pt>20) & (df.jet2_pt>20)]
        df = df[(df.jet1_eta<2.5) & (df.jet1_eta>-2.5)]
        df = df[(df.jet2_eta<2.5) & (df.jet2_eta>-2.5)]
        df = df[(df.jet2_mass>0)] 
        df = df[(df.jet1_mass>0)]
        #df = df[(df.dijet_mass>125-2.5*16.6) & (df.dijet_mass<125+2.5*16.6)]
    
        print("Nan values : %d process %d "%(df.isna().sum().sum(), idx))
        print("Filling jet qgl with 0.5")
        df.jet1_qgl = df.jet1_qgl.fillna(0.5)
        df.jet2_qgl = df.jet2_qgl.fillna(0.5)
        try:
            assert df.isna().sum().sum()==0
            assert df.isna().sum().sum()==0
        except:
            columns_with_nan = df.columns[df.isna().any()].tolist()
            # Find rows with NaN values
            rows_with_nan = df[df.isnull().any(axis=1)]
            print("Columns with NaN values:", columns_with_nan)
            print("\nRows with NaN values:")
            print(rows_with_nan)
            
        print("No Nan values after filling")
        dfs_new.append(df)
    return dfs_new


def preprocess(dfs):
    '''
    dfs is a list of dataframes
    '''
    
    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    dfs_new = []
    for idx, df in enumerate(dfs):
        df = df[(df.jet1_pt>20) & (df.jet2_pt>20)]
        df = df[(df.jet1_eta<2.5) & (df.jet1_eta>-2.5)]
        df = df[(df.jet2_eta<2.5) & (df.jet2_eta>-2.5)]
        df = df[(df.jet2_mass>0)] 
        df = df[(df.jet1_mass>0)]
        df = df[(df.dijet_mass>125-2.5*16.6) & (df.dijet_mass<125+2.5*16.6)]
    
        print("Nan values : %d process %d "%(df.isna().sum().sum(), idx))
        print("Filling jet qgl with 0.5")
        df.jet1_qgl = df.jet1_qgl.fillna(0.5)
        df.jet2_qgl = df.jet2_qgl.fillna(0.5)
        assert df.isna().sum().sum()==0
        assert df.isna().sum().sum()==0
        print("No Nan values after filling")
        dfs_new.append(df)
    return dfs_new


def writeFeatures(featuresForTrainingName, columnsToReadName):
    featuresForTraining=[
       'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
       'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
       'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons',
       'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
       'dijet_pt', 'dijet_eta', 'dijet_phi',
       'dijet_dR',
       'dijet_dEta', 'dijet_dPhi', 'dijet_twist', 'nJets',
       'nJets_20GeV',
       'ht', 'muon_pt', 'muon_eta', 'muon_dxySig', 
       'muon_dzSig', 'muon_IP3d',
       'muon_sIP3d',
       'dijet_cs',
       'muon_pfRelIso03_all', #'muon_tkIsoId'
       #'muon_tightId',
       ]
    
    columnsToRead = [   
        'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons',
       'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
       'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons',
       'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
       'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR',
       'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable', 'dijet_twist', 'nJets',
       'nJets_20GeV',
       'ht', 'muon_pt', 'muon_eta', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d',
       'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_tkIsoId',
       'dijet_cs', 'sf']
    np.save(featuresForTrainingName, featuresForTraining)
    np.save(columnsToReadName, columnsToRead)
    return

def readFeatures(featuresForTrainingName, columnsToReadName):
    featuresForTraining = np.load(featuresForTrainingName)
    columnsToRead = np.load(columnsToReadName)
    return featuresForTraining, columnsToRead



def scale(data, scalerName, fit=False):
    
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            print("feature: %s min: %.1f max: %.1f"%(colName, data[colName].min(), data[colName].max()))
            data[colName] = np.log(1+data[colName])
            print("log done for %s"%colName)
    if fit:
        scaler  = preprocessing.StandardScaler().fit(data[[col for col in data.columns if col!='sf']])
        scaled_array = scaler.transform(data[[col for col in data.columns if col!='sf']])
        scalers = {
            'type'  : 'standard',
            'scaler': scaler,
            }
        with open(scalerName, 'wb') as file:
            pickle.dump(scalers, file)
    else:
        with open(scalerName, 'rb') as file:
            scalers = pickle.load(file)
            scaler = scalers['scaler']
            scaled_array = scaler.transform(data[[col for col in data.columns if col!='sf']])
            
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

    print(dataUnscaled['jet1_pt'])
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            dataUnscaled[colName] = np.exp(dataUnscaled[colName]) - 1
    print(dataUnscaled['jet1_pt'])
    return dataUnscaled