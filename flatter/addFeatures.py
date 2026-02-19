# %%
import numpy as np
import pandas as pd
import glob
import os
from functions import getDfProcesses_v2

# %%
import sys
import argparse




def wrap_delta_phi(dphi):
        """Restores delta phi in [-pi, pi]"""
        dphi = dphi - 2*np.pi*(dphi >= np.pi) + 2*np.pi*(dphi < -np.pi)
        return np.abs(dphi).astype(np.float32)


# %%
def addFeatureToFile(df):
    
    
    # --- 1) Flip z-axis: eta relative to jet1 ---
    leading_eta = df['jet1_eta']
    for col in ['jet1_eta', 'jet2_eta', 'jet3_eta', 'dijet_eta', 'jet1_muon_eta']:
        df[f'{col}_prime'] = df[col] * np.sign(leading_eta)

    # --- 2) Rotate xy-plane: phi relative to jet1 ---
    leading_phi = df['jet1_phi']
    for col in ['jet1_phi', 'jet2_phi', 'jet3_phi', 'dijet_phi', 'jet1_muon_phi']:
        df[f'{col}_prime'] = df[col] - leading_phi
        df[f'{col}_prime'] = wrap_delta_phi(df[f'{col}_prime'])
    
    # --- 3) Change convention of phi: sign of jet2 ---
    subleading_phi = df['jet2_phi_prime']
    for col in ['jet1_phi_prime', 'jet2_phi_prime', 'jet3_phi_prime', 'dijet_phi_prime', 'jet1_muon_phi_prime']:
        df[col] = df[col] * np.sign(subleading_phi)
        df[col] = wrap_delta_phi(df[col])


    scale_of_reference = df['dijet_mass']
    for col in ['jet1_pt', 'jet2_pt', 'jet3_pt',
                'jet1_mass', 'jet2_mass', 'jet3_mass',
                'dijet_pt', 'ht']:
        df[f'{col}_prime'] = df[col]/scale_of_reference

    scale_of_reference_jet1 = df['jet1_pt']
    for col in ['jet1_muon_pt', 'jet1_leadTrackPt']:
        df[f'{col}_prime'] = df[col]/scale_of_reference_jet1

    scale_of_reference_jet2 = df['jet2_pt']
    for col in ['jet2_leadTrackPt']:
        df[f'{col}_prime'] = df[col]/scale_of_reference_jet2
    
    scale_of_reference_jet3 = df['jet3_pt'].replace(0, np.nan)
    for col in ['jet3_leadTrackPt']:
        df[f'{col}_prime'] = df[col]/scale_of_reference_jet3
        df[f'{col}_prime'] = df[f'{col}_prime'].fillna(0)
    return df









# %%
def main():
    parser = argparse.ArgumentParser(description="Script.")
    #### Define arguments
    parser.add_argument("-pN", "--processNumber", type=int, help="processNumber", default=0)

    if 'ipykernel' in sys.modules:
        args = parser.parse_args(args=[])  # Avoids errors when running interactively
    else:
        args = parser.parse_args()
    dfProcesses = getDfProcesses_v2()[0]
    MCList = [
        args.processNumber
    #            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
            #22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42
    #0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
    ]
    
    # %%
    for MCProcess in MCList:
        path = dfProcesses.iloc[MCProcess].flatPath
        print("\nProcess : ",dfProcesses.iloc[MCProcess].process )
        fileNames = glob.glob(path+'/*.parquet')
    # %%



        for idx, file in enumerate(fileNames):
            print(f"{np.round((idx+1)/len(fileNames)*100, 2)} %", end="\r")
            df = pd.read_parquet(file)
            if "jet1_eta_prime" in df.columns:
                #print("Already processed, skipping")
                continue
            df = addFeatureToFile(df)
            if os.path.exists(file):
                os.remove(file)
            df.to_parquet(file)  
        # %%
if __name__=="__main__":
    main()