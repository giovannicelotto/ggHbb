# predict.py
import sys, re
import pandas as pd
import torch
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.scaleUnscale import scale
from helpers.getFeatures import getFeatures
import numpy as np
from functions import getCommonFilters, cut
import joblib
import yaml
from functions import getDfProcesses_v2
import ROOT
import numpy as np
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
DEBUG=False
def apply_jec_and_recompute_root(df_, featuresForTraining, syst_name, direction="up"):
    """
    Apply JEC systematic and recompute all kinematics using TLorentzVector.
    """

    df = df_.copy()
    sign = 1.0 if direction == "up" else -1.0

    initial_HT123_sum = df["jet1_pt_uncor"] + df["jet2_pt_uncor"] + df["jet3_pt"]
    # -----------------------------
    # 1. APPLY JEC TO JET PT
    # -----------------------------
    for i in [1, 2, 3]:
        pt_col = f"jet{i}_pt"
        pt_col_uncor = pt_col+"_uncor"
        syst_col = f"jet{i}_{syst_name.replace('Jet_sys_', '')}"
        #print(syst_col)
        if syst_col in df.columns:
            pass
        else:
            assert False

        if pt_col in df.columns and syst_col in df.columns:
            df[pt_col] = df[pt_col] * (1.0 + sign * df[syst_col])
        if pt_col_uncor in df.columns and syst_col in df.columns:
            df[pt_col_uncor] = df[pt_col_uncor] * (1.0 + sign * df[syst_col])

    # -----------------------------
    # helper
    # -----------------------------
    def make_vec(pt, eta, phi, m):
        v = ROOT.TLorentzVector()
        v.SetPtEtaPhiM(pt, eta, phi, m)
        return v

    # -----------------------------
    # 2. BUILD JETS
    # -----------------------------
    jets = []
    for i in [1, 2, 3]:
        jets.append([
            make_vec(df[f"jet{i}_pt"].iloc[k],
                     df[f"jet{i}_eta"].iloc[k],
                     df[f"jet{i}_phi"].iloc[k],
                     df[f"jet{i}_mass"].iloc[k])
            for k in range(len(df))
        ])
    for i in [1, 2]:
        jets.append([
            make_vec(df[f"jet{i}_pt_uncor"].iloc[k],
                     df[f"jet{i}_eta"].iloc[k],
                     df[f"jet{i}_phi"].iloc[k],
                     df[f"jet{i}_mass"].iloc[k])
            for k in range(len(df))
        ])

    jet1 = jets[0]
    jet2 = jets[1]
    jet3 = jets[2]
    jet1_uncor = jets[3]
    jet2_uncor = jets[4]

    n = len(df)

    # containers
    dijet_pt = np.zeros(n)
    dijet_eta = np.zeros(n)
    dijet_phi = np.zeros(n)
    dijet_mass = np.zeros(n)

    ht = df.ht.copy()

    dR_jet3_dijet = np.zeros(n)
    dPhi_jet3_dijet = np.zeros(n)

    # -----------------------------
    # 3. EVENT LOOP (correct physics)
    # -----------------------------
    for i in range(n):

        v1 = jet1[i]
        v2 = jet2[i]
        v3 = jet3[i]
        v1_uncor = jet1_uncor[i]
        v2_uncor = jet2_uncor[i]

        dijet = v1 + v2

        # dijet observables
        dijet_pt[i] = dijet.Pt()
        dijet_eta[i] = dijet.Eta()
        dijet_phi[i] = dijet.Phi()
        dijet_mass[i] = dijet.M()

        # HT
        ht.iloc[i] = np.float32(    ht.iloc[i] - initial_HT123_sum.iloc[i]    + v1_uncor.Pt() + v2_uncor.Pt() + v3.Pt())

        # angular quantities
        #dEta = v3.Eta() - dijet.Eta()
        dPhi = v3.DeltaPhi(dijet) if v3.Pt()>10 else 0.

        dR_jet3_dijet[i] = v3.DeltaR(dijet) if v3.Pt()>10 else 0.
        dPhi_jet3_dijet[i] = dPhi

    # -----------------------------
    # 4. STORE RESULTS
    # -----------------------------
    df["dijet_pt"] = dijet_pt
    df["dijet_eta"] = dijet_eta
    df["dijet_phi"] = dijet_phi
    df["dijet_mass"] = dijet_mass

    df["ht"] = ht

    df["dR_jet3_dijet"] = dR_jet3_dijet
    df["dPhi_jet3_dijet"] = dPhi_jet3_dijet

    # -----------------------------
    # 5. PRIME VARIABLES
    # -----------------------------
    mjj = dijet_mass + 1e-12

    for i in [1, 2, 3]:
        df[f"jet{i}_pt_prime"] = df[f"jet{i}_pt"] / mjj
        df[f"jet{i}_mass_prime"] = df.get(f"jet{i}_mass", 0) / mjj

    df["dijet_pt_prime"] = df["dijet_pt"] / mjj
    df["ht_prime"] = df["ht"] / mjj

    # -----------------------------
    # 6. MUON IN JET 
    # -----------------------------
    if "jet1_muon_pt" in df.columns:
        df["jet1_muon_pt_prime"] = df["jet1_muon_pt"] / df["jet1_pt"]
    
    
    
    
    def wrap_delta_phi(dphi):
            """Restores delta phi in [-pi, pi]"""
            dphi = dphi - 2*np.pi*(dphi >= np.pi) + 2*np.pi*(dphi < -np.pi)
            return np.abs(dphi).astype(np.float32)
    
    leading_eta = df['jet1_eta']
    for col in ['jet1_eta', 'jet2_eta', 'jet3_eta', 'dijet_eta', 'jet1_muon_eta', 'jet2_muon_eta']:
        df[f'{col}_prime'] = df[col] * np.sign(leading_eta)

    # --- 2) Rotate xy-plane: phi relative to jet1 ---
    leading_phi = df['jet1_phi']
    for col in ['jet1_phi', 'jet2_phi', 'jet3_phi', 'dijet_phi', 'jet1_muon_phi', 'jet2_muon_phi']:
        df[f'{col}_prime'] = df[col] - leading_phi
        df[f'{col}_prime'] = wrap_delta_phi(df[f'{col}_prime'])
    
    # --- 3) Change convention of phi: sign of jet2 ---
    subleading_phi = df['jet2_phi_prime']
    for col in ['jet1_phi_prime', 'jet2_phi_prime', 'jet3_phi_prime', 'dijet_phi_prime', 'jet1_muon_phi_prime', 'jet2_muon_phi_prime']:
        df[col] = df[col] * np.sign(subleading_phi)
        df[col] = wrap_delta_phi(df[col])

    return df

# change the model
# change the features
# change the scaler

def predict(file_path, modelName, boosted, quantile_matching=True, JEC_eval=False, run=2):
    '''
    file_path: path to parquet file (str) or pandas dataframe (pd.DataFrame)
    modelName: name of the model to be used for prediction (str)

    
    '''
    #featuresForTraining, columnsToRead = getFeatures(outFolder=None)
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model"%(modelName)
    featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model/featuresForTraining.npy"%modelName))
    print("Before opening model", flush=True)
    nn = torch.load(modelDir+"/model.pth", map_location=torch.device('cpu'))

    # Load data from file_path and preprocess it as needed
    if type(file_path) == str:
        Xtest = pd.read_parquet(file_path,
                                    engine='pyarrow',
                                     filters= getCommonFilters(btagWP="M", cutDijet=False, ttbarCR='both', boosted=boosted))
    elif isinstance(file_path, pd.DataFrame):
        Xtest = file_path
    

                
    print("File opened", flush=True)

    cols_float = [col for col in Xtest.columns if "topology" not in col]
    data = [Xtest[cols_float]].copy()
    print("Variables recomputed for jec")

    data[0]  = scale(data[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
    for f in data[0].columns:
        print(f,data[0][f].isna().sum())
    nn.eval()
    data_tensor = torch.tensor(np.float32(data[0][featuresForTraining].values)).float()


    if JEC_eval:
        from tqdm import tqdm
        JEC_Variations = [
            "Jet_sys_JECAbsoluteMPFBias", "Jet_sys_JECAbsoluteScale","Jet_sys_JECAbsoluteStat", "Jet_sys_JECFlavorQCD","Jet_sys_JECFragmentation", "Jet_sys_JECPileUpDataMC","Jet_sys_JECPileUpPtBB", "Jet_sys_JECPileUpPtEC1",
            "Jet_sys_JECPileUpPtEC2", "Jet_sys_JECPileUpPtHF","Jet_sys_JECPileUpPtRef", "Jet_sys_JECRelativeBal","Jet_sys_JECRelativeFSR", "Jet_sys_JECRelativeJEREC1","Jet_sys_JECRelativeJEREC2", "Jet_sys_JECRelativeJERHF",
            "Jet_sys_JECRelativePtBB", "Jet_sys_JECRelativePtEC1","Jet_sys_JECRelativePtEC2", "Jet_sys_JECRelativePtHF","Jet_sys_JECRelativeSample", "Jet_sys_JECRelativeStatEC","Jet_sys_JECRelativeStatFSR", "Jet_sys_JECRelativeStatHF",
            "Jet_sys_JECSinglePionECAL", "Jet_sys_JECSinglePionHCAL","Jet_sys_JECTimePtEta",
            "Jet_sys_TotalJECUnc"
        ]

        # IMPORTANT: store original dataframe
        X_nominal = Xtest.copy()

        # container for final systematics (optional)
        jec_outputs = {}

        for JECvar in tqdm(JEC_Variations):

            # UP and DOWN predictions could be stored if needed
            for direction in ["up", "down"]:
                print("JECvar and direction: ", JECvar, direction)
                df_var = apply_jec_and_recompute_root(
                    X_nominal,
                    featuresForTraining,
                    JECvar,
                    direction=direction)
                
                debug_dir = f"/t3home/gcelotto/ggHbb/debug_jec_features/{JECvar}_{direction}"
                os.makedirs(debug_dir, exist_ok=True)
                
                if (DEBUG):
                    for feat in featuresForTraining:

                        nominal_vals = Xtest[feat].replace([np.inf, -np.inf], np.nan).dropna()
                        varied_vals  = df_var[feat].replace([np.inf, -np.inf], np.nan).dropna()

                        if (len(nominal_vals) == 0) or (len(varied_vals) == 0):
                            continue
                        
                        qlow = np.nanpercentile(nominal_vals, 0.5)
                        qhigh = np.nanpercentile(nominal_vals, 99.5)

                        if qlow == qhigh:
                            continue
                        
                        bins = np.linspace(qlow, qhigh, 80)

                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                        ax.hist(
                            nominal_vals,
                            bins=bins,
                            histtype='step',
                            density=True,
                            linewidth=2,
                            label='nominal'
                        )

                        ax.hist(
                            varied_vals,
                            bins=bins,
                            histtype='step',
                            density=True,
                            linewidth=1,
                            label=f'{JECvar}_{direction}'
                        )

                        ax.set_xlabel(feat)
                        ax.set_ylabel("Normalized entries")
                        ax.legend()

                        fig.savefig(f"{debug_dir}/{feat}.png", bbox_inches='tight')

                        plt.close(fig)

                
                # scale with SAME scaler
                df_var = scale(
                    df_var,
                    featuresForTraining=featuresForTraining,
                    scalerName=modelDir + "/myScaler.pkl",
                    fit=False)

                # tensor
                data_tensor = torch.tensor(
                    np.float32(df_var[featuresForTraining].values)
                ).float()

                # prediction
                nn.eval()
                with torch.no_grad():
                    pred = nn(data_tensor).numpy().reshape(-1)
                    if quantile_matching:
                        # Prediction with Quantile Matching + Copula Space
                        qt_tt  = joblib.load("/t3home/gcelotto/ggHbb/tt_CR/analysis/morphing/matrices_and_fitter/qm_old/qt_tt.pkl")
                        qt_ggH = joblib.load("/t3home/gcelotto/ggHbb/tt_CR/analysis/morphing/matrices_and_fitter/qm_old/qt_ggH.pkl")
                        L_tt   = np.load("/t3home/gcelotto/ggHbb/tt_CR/analysis/morphing/matrices_and_fitter/qm_old/L_tt.npy")
                        L_ggH  = np.load("/t3home/gcelotto/ggHbb/tt_CR/analysis/morphing/matrices_and_fitter/qm_old/L_ggH.npy")
                        from copula_morph import copula_morph
                        X_tt_morphed = pd.DataFrame(copula_morph( df_var[featuresForTraining], qt_tt, qt_ggH, L_tt, L_ggH), columns=featuresForTraining, index=data[0].index)
                        quantile_varied_tensor = torch.tensor(np.float32(X_tt_morphed[featuresForTraining].values)).float()
                        data_predictions_qm = nn(quantile_varied_tensor).numpy()
                        jec_outputs[f"qm_{JECvar}_{direction}"] = data_predictions_qm


                jec_outputs[f"{JECvar}_{direction}"] = pred
                jec_outputs[f"dijet_pt_{JECvar}_{direction}"] = df_var["dijet_pt"].values
        return jec_outputs
    
    


    if quantile_matching:

        # Prediction with Quantile Matching + Copula Space
    
        qt_tt  = joblib.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_tt.pkl")
        qt_ggH = joblib.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_ggH.pkl")

        L_tt   = np.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_tt.npy")
        L_ggH  = np.load("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_ggH.npy")

        from copula_morph import copula_morph
        X_tt_morphed = pd.DataFrame(copula_morph( data[0][featuresForTraining], qt_tt, qt_ggH, L_tt, L_ggH), columns=featuresForTraining, index=data[0].index)
        quantile_varied_tensor = torch.tensor(np.float32(X_tt_morphed[featuresForTraining].values)).float()

        with torch.no_grad():  # No need to track gradients for inference
            data_predictions1 = nn(data_tensor).numpy()
            #data_predictions_pca_varied = nn(pca_varied_tensor).numpy()
            data_predictions_qm = nn(quantile_varied_tensor).numpy()
        return data_predictions1, data_predictions_qm
    else:
        with torch.no_grad():
            data_predictions1 = nn(data_tensor).numpy()
        return data_predictions1
    
    


if __name__ == "__main__":
    file_path   = sys.argv[1]
    process        = sys.argv[2]
    modelName        = sys.argv[3]
    boosted        = int(sys.argv[4])
    isMC        = int(sys.argv[5])

    dfProcesses = getDfProcesses_v2()[0]
    JEC_eval = True  if isMC else False
    print("Arguments passed")
    quantile_matching=True
    if quantile_matching:
        predictions1, data_predictions_qm = predict(file_path, modelName, boosted,  quantile_matching=quantile_matching, JEC_eval=False)
        #print("Shape of predictions1", predictions1.shape)
        predictions = pd.DataFrame({
            'PNN': predictions1.reshape(-1),
            #'PNN_pca': data_predictions_pca_varied.reshape(-1),
            'PNN_qm': data_predictions_qm.reshape(-1)
        })
            # -------------------------
            # JEC VARIATIONS
            # -------------------------
        if JEC_eval:
            jec_output = predict(
                file_path, modelName, boosted,
                quantile_matching=True,
                JEC_eval=JEC_eval
            )
        
            # jec_output is assumed dict: {name: array}
            for key, value in jec_output.items():
                print("Adding JEC variation to predictions: ", key)
                predictions[key] = value.reshape(-1)
    else:
        predictions1 = predict(file_path, modelName, boosted, quantile_matching=quantile_matching, JEC_eval=JEC_eval)
        print("Shape of predictions1", predictions1.shape)
        predictions = pd.DataFrame({
            'PNN': predictions1.reshape(-1),
        #    'PNN_pca': data_predictions_pca_varied.reshape(-1),
        #    'PNN_qm': data_predictions_qm.reshape(-1)
        })
    match = re.search(r'_(\d+)\.parquet$', file_path)
    if match:
        number = int(match.group(1))
        print("number matched ", number)
    name = "/scratch/yMjj_%s_FN%d.parquet"%(process, number)
    predictions.to_parquet(name)
    print(name)
