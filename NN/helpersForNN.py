
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd
def preprocessMultiClass(dfs, leptonClass=None, pTmin=None, pTmax=None, suffix=None):
    print(pTmin, pTmax, suffix)
    '''
    dfs is a list of dataframes
    '''

        
    print("Preprocessing...")
    print("Performing the cut in pt and eta")
    print("New function")
    dfs_new = []
    for idx, df in enumerate(dfs):
        #df = df[df.leptonClass == leptonClass]

        print("Initial len df %d : %d"%(idx, len(df)))
        df = df[(df.jet1_pt>20) & (df.jet2_pt>20)]
        df = df[(df.jet2_mass>0)] 
        df = df[(df.jet1_mass>0)]
        if 'jet3_mass' in df.columns:
            df = df[(df.jet3_mass>0)]
        # useless
        df = df[(df.jet1_eta<2.5) & (df.jet1_eta>-2.5)]
        df = df[(df.jet2_eta<2.5) & (df.jet2_eta>-2.5)]
        # end useless
        
        beforeCutMass = len(df)
        df = df[(df.dijet_mass>40) & (df.dijet_mass<200)]
        afterCutMass = len(df)
        print("Eff. cut mass ", afterCutMass/beforeCutMass*100)
        

        print(pTmin, pTmax)
        if (pTmin is not None) & (pTmax is not None):
            print("Pt cut class applied ",pTmin,"-", pTmax)
            print("This is suffix", suffix)
            if (suffix == 'lowPt') | (suffix == 'mediumPt'):
                df = df.loc[ (df.dijet_pt > pTmin) & (df.dijet_pt < pTmax)]
            elif (suffix == 'highPt'):
                df = df.loc[ (df.dijet_pt > pTmin)]
            
            elif ('inclusive' in suffix):
                pass
            
            else:
                assert False
        if df.isna().sum().sum()>0:
            print("Nan values : %d process %d "%(df.isna().sum().sum(), idx))
        print("Filling jet1 qgl with 0. %d" %(df.jet1_qgl.isna().sum()))
        print("Filling jet2 qgl with 0. %d" %(df.jet2_qgl.isna().sum()),"\n")

        df.jet1_qgl = df.jet1_qgl.fillna(0.)
        df.jet2_qgl = df.jet2_qgl.fillna(0.)
        if 'jet3_qgl' in df.columns:
            df.jet3_qgl = df.jet3_qgl.fillna(0.)
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
            
        #print("No Nan values after filling\n")
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



def scale(data, scalerName, fit=False, weights=None):
    
    for colName in data.columns:
        if ("_pt" in colName) | ("_mass" in colName) | (colName=="ht"):
            print("feature: %s min: %.1f max: %.1f"%(colName, data[colName].min(), data[colName].max()))
            data[colName] = np.log(1+data[colName])
            print("log done for %s"%colName)
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