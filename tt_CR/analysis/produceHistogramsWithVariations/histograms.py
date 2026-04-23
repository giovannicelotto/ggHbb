import ROOT

def make_histograms(mc_variations, dfData, config, category, n_bins):
    fout = ROOT.TFile(
        f"/t3home/gcelotto/ggHbb/tt_CR/histograms/histograms_{category}.root",
        "RECREATE"
    )
    for variation, dfMC in mc_variations.items():

        suffix = "" if variation == "nominal" else f"_{variation}"




        x_min = config["lower_NN"]
        x_max = config["upper_NN"]

        for proc in dfMC["process"].unique():

            if proc in ["ggH(bb)", "VBFH(bb)"]:
                continue

            hist = ROOT.TH1F(f"h_{proc}{suffix}", f"h_{proc}{suffix}", n_bins, x_min, x_max)
            hist.Sumw2()

            df_sel = dfMC[
                (dfMC["process"] == proc) &
                (dfMC.is_ttbar_CR == 1) &
                (dfMC.PNN_qm >= x_min) &
                (dfMC.PNN_qm < x_max)
            ]

            for x, w in zip(df_sel["PNN_qm"], df_sel["weight"]):
                hist.Fill(float(x), float(w))

            hist.Write()

        # data
        if variation != "nominal":
            continue
        hist_data = ROOT.TH1F("data_obs", "data_obs", n_bins, x_min, x_max)

        df_data_sel = dfData[
            (dfData.is_ttbar_CR == 1) &
            (dfData.PNN_qm >= x_min) &
            (dfData.PNN_qm < x_max)
        ]

        for x in df_data_sel["PNN_qm"]:
            hist_data.Fill(float(x))

        hist_data.Write()
    fout.Close()