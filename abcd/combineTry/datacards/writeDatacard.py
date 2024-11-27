# Define the variables
channels = [
        "lc1_cs0",
        "lc1_cs1",
        "lc2_cs0",
        "lc2_cs1",
        "lc3_cs0",
        "lc3_cs1",
        "total"
    ]
for channel in channels:
    datacard_name = "/t3home/gcelotto/ggHbb/abcd/combineTry/datacards/shapeZdatacard_%s.txt"%channel
    bin_name = channel
    processes = ["ZJets", "H", "VV", "ST", "ttbar", "WJets", "QCD"]
    process_indices = [0, 1, 2, 3, 4, 5, 6]
    rates = [-1, -1, -1, -1, -1, -1, -1]
    nuisances = [
        ("Z_xsec", [1.05, "-", "-", "-", "-", "-", "-"]),
        ("H_xsec", ["-", 1.05, "-", "-", "-", "-", "-"]),
        ("VV_xsec", ["-", "-", 1.05, "-", "-", "-", "-"]),
        ("ST_xsec", ["-", "-", "-", 1.05, "-", "-", "-"]),
        ("ttbar_xsec", ["-", "-", "-", "-", 1.05, "-", "-"]),
        ("W_xsec", ["-", "-", "-", "-", "-", 1.05, "-"]),
        ("lumi", [1.025, 1.025, 1.025, 1.025, 1.025, 1.025, "-"]),
        ("QCD_closure", ["-", "-", "-", "-", "-", "-", 1.05]),
    ]

    shapes_file = "/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_%s.root"%channel

    # Write the datacard
    with open(datacard_name, "w") as file:
        #file.write("Combination of datacard_Zxsection.txt\n")
        file.write("imax 1 number of bins\n")
        file.write("jmax 6 number of processes minus 1\n")
        file.write("kmax 8 number of nuisance parameters\n")
        file.write("-" * 130 + "\n")
        file.write(f"shapes *    {bin_name}  {shapes_file} $PROCESS\n")
        file.write("-" * 130 + "\n")
        file.write(f"bin          {bin_name}\n")
        file.write("observation  -1\n")
        file.write("-" * 130 + "\n")

        # Write bin and process headers
        file.write(f"bin                             {'     '.join([bin_name] * len(processes))}\n")
        file.write(f"process                         {'     '.join(processes)}\n")
        file.write(f"process                         {'     '.join(map(str, process_indices))}\n")
        file.write(f"rate                            {'     '.join(map(str, rates))}\n")
        file.write("-" * 130 + "\n")

        # Write nuisance parameters
        for nuisance, values in nuisances:
            file.write(f"{nuisance:<24} lnN     {'     '.join(map(str, values))}\n")

        file.write("-" * 130 + "\n")
        file.write("syst      group = Z_xsec H_xsec VV_xsec ST_xsec ttbar_xsec W_xsec lumi QCD_closure")
