import sys
sys.path.append("/t3home/gcelotto/ggHbb/WSFit/allSteps/helpers")
from getDfsFromConfig import extract_pnn_edges
def load_config(category):
    import yaml

    # cuts
    config_path_cuts = f"/t3home/gcelotto/ggHbb/WSFit/Configs/cat{int(category)}.yml"
    with open(config_path_cuts, 'r') as f:
        config_cuts = yaml.safe_load(f)

    lower_NN, upper_NN = extract_pnn_edges(config_cuts["cuts_string"])

    # main cfg
    cfg_file = "/t3home/gcelotto/ggHbb/tt_CR/analysis/plot_tt_from_df.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    config_cuts["cuts_string"] = config_cuts["cuts_string"].replace("PNN", "PNN_qm")

    return {
        "cuts": config_cuts["cuts_string"],
        "lower_NN": lower_NN,
        "upper_NN": upper_NN,
        "modelName": cfg["modelName"],
        "columns": cfg["columns"],
        "MConlyFeatures": cfg["MConlyFeatures"],
        "dataPeriods": cfg["dataPeriods"],
        "MC": cfg["MC"],
        "df_folder": f"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/{cfg['modelName']}",
        "systematics": cfg.get("systematics", [])
  
  
    }