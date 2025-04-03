
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
# %%
class FitWithSystematics:
    def __init__(self, model_name, path, data_frame_list, x1, x2, out_folder, fitFunction):
        self.model_name = model_name
        self.path = path
        self.fitFunction = fitFunction
        self.data_frame_list = data_frame_list
        self.x1, self.x2 = x1, x2
        self.out_folder = out_folder
        self.parameters = {}
        # Map variation names to lambda functions that modify the weights
        self.variations_map = {
            "jet1_btag_up": lambda df: df['weight'] * df['jet1_btag_up']/df['jet1_btag_central'],
            "jet1_btag_down": lambda df: df['weight'] * df['jet1_btag_down']/df['jet1_btag_central'],
            # Add more variations as needed
        }

    def load_data(self):
        # Load the MC datasets
        dfs_mc = []
        for process_name in self.data_frame_list.process.values:
            print(f"Opening {process_name}")
            df = pd.read_parquet(f"{self.path}/df_{process_name}_{self.model_name}.parquet")
            dfs_mc.append(df)
        return dfs_mc

    def apply_cuts(self, dfs_mc, cuts_dict):
        # Apply cuts based on the cuts_dict (this dict is specific to a category)
        print(f"Applying cuts: {cuts_dict}")
        for feature, (min_val, max_val) in cuts_dict.items():
            print(f" - Cutting feature {feature}: {min_val} < {feature} < {max_val}")
            dfs_mc = cut(dfs_mc, feature, min_val, max_val)
        return dfs_mc

    def fit_model(self, x, cTot, err, fitregion):
        if self.fitFunction=='zPeak_dscb':
            least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], zPeak_dscb)
            m = Minuit(least_squares,
                       normSig=cTot.sum() * (x[1] - x[0]),
                       fraction_dscb=0.55,
                       mean=92.61,
                       sigma=10.6,
                       alphaL=0.89,
                       nL=8,
                       alphaR=1.77,
                       nR=0.58,
                       sigmaG=10.9)
            m.print_level = 2
            m.limits['fraction_dscb'] = (0.05, 0.95)
            m.limits['mean'] = (83, 97)
            m.limits['nR'] = (1e-7, 3)
            m.limits['nL'] = (1e-12, 300)
            m.limits['sigma'] = (5, 15)
            m.limits['sigmaG'] = (5, 15)
            m.errors["alphaL"] = 0.2
            m.errors["alphaR"] = 0.1
            m.errors["sigma"] = 1
            m.errors["nR"] = 0.1
            m.errors["nL"] = 100
            m.errors["normSig"] = 50
        elif self.fitFunction=='zPeak_rscb':
            least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], zPeak_rscb)
            m = Minuit(least_squares,
                        normSig=cTot.sum() * (x[1] - x[0]),
                    fraction_dscb=0.9,
                        mean=92.82,
                        sigma=17.6,
                        alphaR=1.97,
                        nR=0.65,
                        sigmaG=9,
                        )
            m.print_level=2
            m.limits['fraction_dscb'] = (0., 0.95)

        m.migrad(ncall=2000, iterate=50)
        return m

    def save_parameters(self, variation_name, m):
        fit_params = {name: {"value": m.values[name], "error": m.errors[name]} for name in m.parameters}
        self.parameters[variation_name] = fit_params


    def apply_variations(self, variations, dfs_mc, bins, fitregion):
        # Apply variations and fit for each case
        for variation_name in variations:
            # Apply variation to the data (affects weights)
            varied_dfs = self.apply_variation_to_data(dfs_mc, variation_name)
            
            # Recompute histograms after variation
            cTot = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            for df in varied_dfs:
                c = np.histogram(df.dijet_mass, bins=bins, weights=df.weight_)[0]
                cerr = np.histogram(df.dijet_mass, bins=bins, weights=(df.weight_)**2)[0]
                err += cerr
                cTot += c
            err = np.sqrt(err)
            
            # Fit the varied model and save parameters
            m = self.fit_model((bins[1:] + bins[:-1]) / 2, cTot, err, fitregion)
            self.save_parameters(variation_name, m)


    def apply_variation_to_data(self, dfs_mc, variation_name):
        if variation_name not in self.variations_map:
            print(f"Variation {variation_name} is not recognized!")
            return dfs_mc
        
        variation_func = self.variations_map[variation_name]
        for df in dfs_mc:
            df['weight_'] = variation_func(df)  # Apply the corresponding variation function

        return dfs_mc

    def save_results(self):
        # Save the parameters to a JSON file
        with open(f"{self.out_folder}/fit_parameters_with_systematics.json", "w") as f:
            json.dump({'fitFunction': self.fitFunction, 'parameters': self.parameters}, f, indent=4)


    def plot_results(self, x, cTot, err, m, fitregion, bins, out_folder):
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(x, cTot, err, marker='o', color='black', linestyle='none')

        x_draw = np.linspace(self.x1, self.x2, 1001)

        if self.fitFunction=='zPeak_dscb':
            y_draw = zPeak_dscb(x_draw, *[m.values[p] for p in m.parameters])
            y_values = zPeak_dscb(x, *[m.values[p] for p in m.parameters])
            label = 'DSCB + Gaus Fit'
        elif self.fitFunction=='zPeak_rscb':
            y_draw = zPeak_rscb(x_draw, *[m.values[p] for p in m.parameters])
            y_values = zPeak_rscb(x, *[m.values[p] for p in m.parameters])
            label = 'RSCB + Gaus Fit'

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
            if self.fitFunction=='zPeak_dscb':
                y_draw = zPeak_dscb(x_draw, *values)
            elif self.fitFunction=='zPeak_rscb':
                y_draw = zPeak_rscb(x_draw, *values)
            ax.plot(x_draw, y_draw, label=f'Fit - {variation_name}', linewidth=1)

        # Add a legend and labels
        ax.legend()
        ax.set_xlabel('Dijet Mass (GeV)')
        ax.set_ylabel('Events')

        # Save the plot
        fig.savefig(f"{out_folder}/plots/zPeakFit_with_systematics_variations.png", bbox_inches='tight')


# %%
