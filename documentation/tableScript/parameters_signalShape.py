# %%
import json
import pandas as pd

categories = [0,10,100]
categoriesLabels = ["LL", "MM", "TT"]

# %%
# %%
for particle in ["Z", "H"]:

    categories = [0, 10, 100]
    categoriesLabels = ["LL", "MM", "TT"]

    rows = []

    for cat, label in zip(categories, categoriesLabels):
        path = f"/t3home/gcelotto/ggHbb/WSFit/output/cat{cat}/fit_parameters_with_systematics_{particle}.json"
        with open(path, 'r') as f:
            parameters = json.load(f)

        pars = parameters['parameters']['nominal']

        rows.append({
            "Category": label,
            r"$f_{\mathrm{G}}$": (1 - pars['fraction_dscb']['value'], pars['fraction_dscb']['error']),
            r"$\mu$": (pars['mean']['value'], pars['mean']['error']),
            r"$\sigma_{\mathrm{CB}}$": (pars['sigma']['value'], pars['sigma']['error']),
            r"$\alpha_L$": (pars['alphaL']['value'], pars['alphaL']['error']),
            r"$n_L$": (pars['nL']['value'], pars['nL']['error']),
            r"$\alpha_R$": (pars['alphaR']['value'], pars['alphaR']['error']),
            r"$n_R$": (pars['nR']['value'], pars['nR']['error']),
            r"$\sigma_{\mathrm{G}}$": (pars['sigmaG']['value'], pars['sigmaG']['error']),
        })

    # Build LaTeX manually (cleanest, no pandas tricks)
    latex = r"""
    \begin{table}[htbp]
    \centering
    \caption{Fit parameters of the Gaussian + Double-Sided Crystal Ball model for \%sbb.}
    \label{tab:fit_parameters_all}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{l|cc|cc|cc}
    \hline
    & \multicolumn{6}{c}{Category} \\
    Parameter
    & \multicolumn{2}{c}{LL}
    & \multicolumn{2}{c}{MM}
    & \multicolumn{2}{c}{TT} \\
    & Value & Unc. & Value & Unc. & Value & Unc. \\
    \hline
    """%particle.lower()

    for par in rows[0].keys():
        if par == "Category":
            continue
        latex += f"{par}"
        for r in rows:
            v, e = r[par]
            latex += f" & {v:.4g} & {e:.4g}"
        latex += r" \\" + "\n"

    latex += r"""
    \hline
    \end{tabular}
    \end{table}
    """

    print(latex)

    with open(f"/t3home/gcelotto/ggHbb/documentation/plots/{particle}_fit_parameters_table_combined.txt", "w") as f:
            f.write(latex)


# %%
assert False, "Stop here"
for cat, label in zip(categories, categoriesLabels):
    path = "/t3home/gcelotto/ggHbb/WSFit/output/cat%d/fit_parameters_with_systematics_Z.json"%cat
    with open(path, 'r') as f:
        parameters = json.load(f)

    pars = parameters['parameters']['nominal']



    fraction_G = 1 - pars['fraction_dscb']['value']
    fraction_G_err = pars['fraction_dscb']['error']

    rows = [
        (r"$f_{\mathrm{G}}$", fraction_G, fraction_G_err),
        (r"$\mu$", pars['mean']['value'], pars['mean']['error']),
        (r"$\sigma_{\mathrm{CB}}$", pars['sigma']['value'], pars['sigma']['error']),
        (r"$\alpha_L$", pars['alphaL']['value'], pars['alphaL']['error']),
        (r"$n_L$", pars['nL']['value'], pars['nL']['error']),
        (r"$\alpha_R$", pars['alphaR']['value'], pars['alphaR']['error']),
        (r"$n_R$", pars['nR']['value'], pars['nR']['error']),
        (r"$\sigma_{\mathrm{G}}$", pars['sigmaG']['value'], pars['sigmaG']['error']),
    ]

    df = pd.DataFrame(rows, columns=["Parameter", "Value", "Uncertainty"])

    latex_table = df.to_latex(
        index=False,
        float_format="%.4g",
        escape=False,
        column_format="lcc",
        caption="Fit parameters of the Gaussian + Double-Sided Crystal Ball model for category %s"%label,
        label="tab:fit_parameters",

    )
    latex_table_fixed = latex_table.replace(r'\toprule', r'\hline') \
                               .replace(r'\midrule', r'\hline') \
                               .replace(r'\bottomrule', r'\hline')
    print(latex_table_fixed)

    # Save the LaTeX table to a .txt file
    with open("/t3home/gcelotto/ggHbb/documentation/plots/fit_parameters_table_%s.txt"%label, "w") as f:
        f.write(latex_table_fixed)

    # %%