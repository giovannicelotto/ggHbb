import argparse
from pathlib import Path
import yaml
import uproot
def generate_datacard(
    output_path,
    ws_file,
    ws_name,
    cat_name,
    processes,
    SF_NN,
    SF_err,
    signal_name="signal",
    background_name="background",
    lumi=1.025,
):
    #with open(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat_name}_Hsyst.yaml", "r") as f:
    #    Hsyst = yaml.safe_load(f) or {}
    #with open(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat_name}_Zsyst.yaml", "r") as f:
    #    Zsyst = yaml.safe_load(f) or {}
    """
    Generates a datacard txt file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Assign process indices: signal = 0, others in order
    process_indices = [0] + list(range(1, len(processes)+2))
    
    # Build process and rate lines
    process_line = "process      " + " ".join([signal_name] + [background_name] + processes)
    index_line = "process      " + " ".join(map(str, process_indices))
    rate_line = "rate         " + " ".join(["1"] * (len(processes) + 2))
    
    datacard = f"""# Datacard example for H->gg like bias study
imax 1
jmax {len(processes)+1}
kmax *
---------------------------------------------
shapes data_obs    * {ws_file} {ws_name}:rooHist_data_cat{cat_name}
shapes {signal_name}      * {ws_file} {ws_name}:model_H_c{cat_name} {ws_name}:model_H_c{cat_name}_$SYSTEMATIC
"""
    # add shapes for other processes
    for proc in processes:
        datacard += f"shapes {proc}      * {ws_file} {ws_name}:model_{proc}_c{cat_name} {ws_name}:model_{proc}_c{cat_name}_$SYSTEMATIC\n"
    
    # background
    datacard += f"shapes {background_name}  * {ws_file} {ws_name}:CMS_hgg_0_2016_13TeV_bkg_noZ\n"
    
    datacard += f"""---------------------------------------------
bin Cat{cat_name}
observation -1
---------------------------------------------
bin          {" ".join(["Cat"+cat_name]*(len(processes)+2))}
{process_line}
{index_line}
{rate_line}
---------------------------------------------
lumi lnN {lumi}  - {" ".join([str(lumi)]*len(processes))}
---------------------------------------------
rateZbb  rateParam  Cat{cat_name}  Z  1.0  [-1.0,3.0]
SF_NN_{cat_name}   rateParam  Cat{cat_name}  signal  {SF_NN}
SF_NN_{cat_name}   rateParam  Cat{cat_name}  Z       {SF_NN}
SF_NN_{cat_name}   param      {SF_NN}  {SF_err}
---------------------------------------------
"""
    
    # Optional: you can add rateParam or other discrete lines as needed
    datacard += f"pdfindex_{cat_name}_2016_13TeV discrete\n"
    #datacard += f"puid   shape 1 - 1\n"
    #datacard += f"btag_hf   shape 1 - 1\n"
    #datacard += f"btag_lightf   shape 1 - 1\n"
    datacard += f"---------------------------------------------\n"
    
    
    # Write to file
    with open(output_path, "w") as f:
        f.write(datacard)
    
    print(f"Datacard written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datacards")
    parser.add_argument("--output", required=True, help="Output datacard path")
    parser.add_argument("--ws_file", required=True, help="Workspace root file")
    parser.add_argument("--ws_name", default="ws3", help="Workspace object name")
    parser.add_argument("--cat", default="Cat1", help="Category name")
    parser.add_argument("--processes", nargs="+", default=["Z"], help="List of processes")
    parser.add_argument("--lumi", type=float, default=1.025, help="Luminosity uncertainty")





    
    args = parser.parse_args()
    location_of_fit = "/t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm"
    f = uproot.open("%s/fitDiagnosticscat%s.root"%(location_of_fit, args.cat))
    SF_NN = f["tree_fit_sb"].arrays()["SF_NN"][0]
    SF_NNLoErr = f["tree_fit_sb"].arrays()["SF_NNLoErr"][0]
    SF_NNHiErr = f["tree_fit_sb"].arrays()["SF_NNHiErr"][0]
    #print(SF_NNLoErr, SF_NNHiErr)
    SF_err_symm = max(abs(SF_NNLoErr), abs(SF_NNHiErr))
    
    generate_datacard(
        output_path=args.output,
        ws_file=args.ws_file,
        ws_name=args.ws_name,
        cat_name=args.cat,
        processes=args.processes,
        SF_NN=SF_NN,
        SF_err=SF_err_symm,
        lumi=args.lumi,
    )



#alphaS   lnN {Hsyst['alphaS']} - {Zsyst['alphaS']}\n
#PS_ISR   lnN {Hsyst['PS_ISR']} - {Zsyst['PS_ISR']}\n
#PS_FSR   lnN {Hsyst['PS_FSR']} - {Zsyst['PS_FSR']}\n
#Scale   lnN {Hsyst['Scale']} - {Zsyst['Scale']}\n