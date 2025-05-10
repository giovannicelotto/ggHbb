
# %%
import numpy as np
import json, sys
from iminuit import Minuit
from iminuit.cost import LeastSquares
import pandas as pd
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.allFunctions import *
import matplotlib.pyplot as plt
import mplhep as hep
from functions import cut
hep.style.use("CMS")
from scipy.stats import chi2
import math

def dijet_mass_inline(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    def to_four_vector(pt, eta, phi, mass):
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        E = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
        return (E, px, py, pz)

    E1, px1, py1, pz1 = to_four_vector(pt1, eta1, phi1, mass1)
    E2, px2, py2, pz2 = to_four_vector(pt2, eta2, phi2, mass2)

    E_total = E1 + E2
    px_total = px1 + px2
    py_total = py1 + py2
    pz_total = pz1 + pz2

    mass2 = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    return math.sqrt(mass2) if mass2 > 0 else 0.0

def apply_dijet_variation(df, var_name, direction):
    sign = +1 if direction == "Up" else -1
    var1_col = f"jet1_sys_{var_name}_Up"
    var2_col = f"jet2_sys_{var_name}_Up"

    return df.apply(
        lambda row: dijet_mass_inline(
            row.jet1_pt * (1 + sign * row[var1_col]),
            row.jet1_eta,
            row.jet1_phi,
            row.jet1_mass * (1 + sign * row[var1_col]),
            row.jet2_pt * (1 + sign * row[var2_col]),
            row.jet2_eta,
            row.jet2_phi,
            row.jet2_mass * (1 + sign * row[var2_col]),
        ), axis=1
    )
# %%
class FitWithSystematics:
    def __init__(self, model_name, path, dfProcesses, x1, x2, out_folder, fitFunction, dfProcesses_sD, dfProcesses_sU):
        self.model_name = model_name                    # to load data
        self.path = path                                # To load data
        self.fitFunction = fitFunction                  # Function to fit Z peak
        self.dfProcesses = dfProcesses          # list of processes to Z
        self.dfProcesses_sD = dfProcesses_sD          # list of processes to Z smearing Down
        self.dfProcesses_sU = dfProcesses_sU          # list of processes to Z smearing up
        self.x1, self.x2 = x1, x2                       # boundaries
        self.out_folder = out_folder                    # outfolder for plots
        self.parameters = {}
        # Map variation names to lambda functions that modify the weights
        self.variations_map = {
            "btag_up": lambda df: df['weight'] * df['btag_up']/df['btag_central'],
            "btag_down": lambda df: df['weight'] * df['btag_down']/df['btag_central'],}
        #self.variations_dijet_map = {
        #        f"{var}_{direction}": lambda df, v=var, d=direction: apply_dijet_variation(df, v, d)
        #        for var in [                    'JECAbsoluteMPFBias','JECAbsoluteScale','JECAbsoluteStat','JECFlavorQCD','JECFragmentation','JECPileUpDataMC','JECPileUpPtBB','JECPileUpPtEC1','JECPileUpPtEC2','JECPileUpPtHF','JECPileUpPtRef','JECRelativeBal','JECRelativeFSR','JECRelativeJEREC1','JECRelativeJEREC2','JECRelativeJERHF','JECRelativePtBB','JECRelativePtEC1','JECRelativePtEC2','JECRelativePtHF','JECRelativeSample','JECRelativeStatEC','JECRelativeStatFSR','JECRelativeStatHF','JECSinglePionECAL','JECSinglePionHCAL','JECTimePtEta',]
        #        for direction in ["Up", "Down"]
        #}

    def load_data(self, smear=0, varJEC=None):
        # Load the MC datasets
        if smear == -1:
            process_list = self.dfProcesses_sD
        elif smear == 1:
            process_list = self.dfProcesses_sU
        else:
            process_list = self.dfProcesses

        dfs_mc = []
        if varJEC==None:
            for process_name in process_list.process.values:
                print(f"Opening {process_name}")
                print(f"{self.path}/df_{process_name}_{self.model_name}.parquet")
                df = pd.read_parquet(f"{self.path}/df_{process_name}_{self.model_name}.parquet")
                dfs_mc.append(df)
        else:
            print("Here we are in JEC")
            for process_name in process_list.process.values:
                print(f"Opening {process_name}")
                print(f"{self.path}/df_{process_name}_{varJEC}_{self.model_name}.parquet")
                df = pd.read_parquet(f"{self.path}/df_{process_name}_{varJEC}_{self.model_name}.parquet")
                dfs_mc.append(df)
            



        print("Sum of weights")
        print(dfs_mc[0].weight.sum())
        print(len(dfs_mc[0]))
        return dfs_mc

    def apply_cuts(self, dfs_mc, cuts_dict):
        # Apply cuts based on the cuts_dict (this dict is specific to a category)
        print(f"Applying cuts: {cuts_dict}")
        for feature, (min_val, max_val) in cuts_dict.items():
            print(f" - Cutting feature {feature}: {min_val} < {feature} < {max_val}")
            dfs_mc = cut(dfs_mc, feature, min_val, max_val)
        return dfs_mc

    def fit_model(self, x, cTot, err, fitregion, params, paramsLimits):
        params["normSig"] = cTot.sum() * (x[1] - x[0])
        fit_func = globals()[self.fitFunction]
        least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], fit_func)
        
        m = Minuit(least_squares,
                    **params
                    )
        m.print_level = 0
        for par in m.parameters:
            if par in paramsLimits:
                m.limits[par] = paramsLimits[par]  # Assign limits from the dictionary
                m.print_level=0

        m.migrad(ncall=2000, iterate=50)
        print( "%.1f/%d"%(m.fval, len(x[fitregion]) - m.nfit))
        return m

    def save_parameters(self, variation_name, m):
        fit_params = {name: {"value": m.values[name], "error": m.errors[name]} for name in m.parameters}
        self.parameters[variation_name] = fit_params


    def apply_variations(self, variations, dfs_mc, bins, fitregion, params, paramsLimits, cuts_dict):
        '''
        variations : list of string of variations
        dfsMC : list of dataframes of Z in the nominal case
        bins : np.array
        fitregion :
        params:
        paramsLimits
        
        '''
        # Apply variations and fit for each case
        for variation_name in variations:
            print(variation_name)
            # Apply variation to the data (affects weights)
            if variation_name == "JER_Up":
                varied_dfs = self.load_data(smear=1)
                varied_dfs = self.apply_cuts(varied_dfs, cuts_dict)
                for df in varied_dfs:
                    df['weight_'] = df['weight']
                    df['dijet_mass_'] = df['dijet_mass']
            elif variation_name == "JER_Down":
                varied_dfs = self.load_data(smear=-1)
                varied_dfs = self.apply_cuts(varied_dfs, cuts_dict)
                for df in varied_dfs:
                    df['weight_'] = df['weight']
                    df['dijet_mass_'] = df['dijet_mass']
            elif (variation_name=='btag_up') | (variation_name=='btag_down'):
                varied_dfs = self.apply_variation_to_data(dfs_mc, variation_name)
            elif "JEC" in variation_name:
                varied_dfs = self.load_data(varJEC=variation_name, smear=0)
                varied_dfs = self.apply_cuts(varied_dfs, cuts_dict)
                for df in varied_dfs:
                    df['weight_'] = df['weight']
                    df['dijet_mass_'] = df['dijet_mass']
            
            # Recompute histograms after variation
            cTot = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            for df in varied_dfs:
                c = np.histogram(df.dijet_mass_, bins=bins, weights=df.weight_)[0]
                cerr = np.histogram(df.dijet_mass_, bins=bins, weights=(df.weight_)**2)[0]
                err += cerr
                cTot += c
            #print(cTot)
            err = np.sqrt(err)
            
            # Fit the varied model and save parameters
            m = self.fit_model((bins[1:] + bins[:-1]) / 2, cTot, err, fitregion, params, paramsLimits)
            self.save_parameters(variation_name, m)


    def apply_variation_to_data(self, dfs_mc, variation_name):
        #if variation_name not in self.variations_map:
        #    print(f"Variation {variation_name} is not recognized!")
        #    return dfs_mc
        
        for df in dfs_mc:
            #if 'JEC' in variation_name:
            #    variation_func_dijetMass = self.variations_dijet_map[variation_name]
            #    df['dijet_mass_'] = variation_func_dijetMass(df)  # Apply the corresponding variation function
            #    df['weight_'] = df.weight
            variation_func = self.variations_map[variation_name]
            df['weight_'] = variation_func(df)  # Apply the corresponding variation function
            df['dijet_mass_'] = df.dijet_mass 

        return dfs_mc

    def save_results(self):
        # Save the parameters to a JSON file
        with open(f"{self.out_folder}/fit_parameters_with_systematics.json", "w") as f:
            json.dump({'fitFunction': self.fitFunction, 'parameters': self.parameters}, f, indent=4)


    def plot_results(self, x, cTot, err, m, fitregion, bins, out_folder):
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(x, cTot, err, marker='o', color='black', linestyle='none')

        x_draw = np.linspace(self.x1, self.x2, 1001)

        fit_func = globals()[self.fitFunction]
        params = [m.values[p] for p in m.parameters]

        y_draw = fit_func(x_draw, *params)
        y_values = fit_func(x, *params)

        label_map = {
            'zPeak_dscb': 'DSCB + Gaus Fit',
            'zPeak_rscb': 'RSCB + Gaus Fit',
            'zPeak_rscb_pol1': 'RSCB + Gaus Fit + pol1'
        }
        label = label_map.get(self.fitFunction, 'Unknown Fit')

        chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
        ndof = len(x[fitregion]) - len(m.parameters)
        chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
        ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')
        
        ax.plot(x_draw, y_draw, label=label, color='red', linewidth=1)
        ax.legend()
        ax.set_ylim(0, ax.get_ylim()[1])
        fig.savefig(f"{out_folder}/plots/zPeakFit_with_systematics.png", bbox_inches='tight')
    def plot_variations_results(self, x, cTot, err, bins, fitregion, out_folder):
        fig, ax = plt.subplots(1, 1)

        # Plot the original data
        ax.errorbar(x, cTot, err, marker='o', color='black', linestyle='none', label='MC nominal')

        # Draw the fitted curves for each variation
        x_draw = np.linspace(self.x1, self.x2, 1001)
        for variation_name, fit_params in self.parameters.items():
            # Retrieve the parameters for the variation
            values = [fit_params[param]["value"] for param in fit_params]
            fit_func = globals()[self.fitFunction]
            y_draw = fit_func(x_draw, *values)
            ax.plot(x_draw, y_draw, label=f'Fit - {variation_name}', linewidth=1)

        # Add a legend and labels
        ax.legend()
        ax.set_xlabel('Dijet Mass (GeV)')
        ax.set_ylabel('Events')

        # Save the plot
        fig.savefig(f"{out_folder}/plots/zPeakFit_with_systematics_variations.png", bbox_inches='tight')


# %%
