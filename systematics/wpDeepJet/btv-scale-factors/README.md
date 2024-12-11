# btv-scale-factors

Centrally managed collection of all scale factors provided to BTV.

## SF status (Run 3)

- [Summer22](2022_Summer22/README.md)
- [Summer22EE](2022_Summer22EE/README.md)
- [Summer23](2023_Summer23/README.md)
- [Summer23BPix](2023_Summer23BPix/README.md)

## Instructions (Run3)

For Run3 the format of providing csv files has to be homogenized, hence only csv files adhereing to a certain standard are accepted.
Please check, before setting up a merge request, whether the csv files provided adhere to the new standard. You can do this by checking this with the `test/validate_csv.py` script, e.g.
```bash
python3 test/validate_csv.py path/to/new/csv/file.csv
```
This script also automatically runs as a CI pipeline after pushing/merging, and will flag all ill-formatted files.   

An example of a well-formatted csv file can be found here: [2022_test/csv/btagging_fixedWP_SFb/particleNet_simpleSyst_example_v1.csv](2022_test/csv/btagging_fixedWP_SFb/pNet_simpleSyst_example_v1.csv).

#### CSV file content

The csv files should be homogeneously readable with `pandas`, i.e. should be csv-parseable and contain homogeneous column names and column entries.
A checklist of things to consider:
- The column names should be `wp`,`type`,`syst`,`flav`,`etaMin/Max`,`ptMin/Max`,`discrMin/Max`,`formula`
- There should be no spaces between entries or column headers
- The `wp` column should only contain the values `L/M/T/XT/XXT/-`, where `-` is for the shape correction
- The `type` column should only contain values such as `comb/mujets/light/shape/ptrel/sys8/ltsv/tnp/kinfit/wc/negtag`
- The `syst` column should either be `central` or start with `up/down_XXX`
- The `flav` column should only contain the values `0/4/5` indicating light/charm/b jets
- The `eta/pt/discr` columns should not contain negative values, this is especially important for the `eta` columns, which should in principle be defined as `abs(eta)`, i.e. no negative values.
- The `formula` column doesnt have any restrictions so far


#### CSV file naming and locations

The csv files should be placed according to a set naming convention:
- For each SF campaign one directory exists, e.g. `2022_Summer22EE`, in this directory is a `csv` subdirectory, containing more directories for the different SF types.
- The subdirectories in the `csv` directories should follow the naming convention of `(b/c)tagging_(fixedWP/shape)_(SFb/SFc/SFlight/itFit)`.
- The csv files in these directories should follow the naming convention `[taggerName]_[descriptor]_v*.csv`.
- The `taggerName` should be one of the following: `deepJet/particleNet/robustParticleTransformer`.
- The `descriptor` should be something useful, like the SF method `ptrel/sys8/ltsv/tnp/kinfit/wc/negtag`, or for the combination the uncertainty breakdown `fullBreakdown/yearCorrelation/simple`, or for the shape correction the JEC scheme, e.g. `jesTotal/jesReduced/jesFull`.
- The csv file name has to end with a versioning number, i.e. starting from `v1`, and increasing if changes are made.

## Old Instructions (Run2)

Please provide the csv files in the directories for the different data eras. Add a subdirectory for each SF method and make sure the naming of sub directories and csv files is homogeneous between the different eras.

This could for example look like this:
```
UL2018/
    btagging_ptRel/
        deepCSVXXXX.csv
        deepJetXXXX.csv
    subjet_tagging/
        deepCSVXXXX.csv
    ctagging_wcharm/
        deepCSVXXXX.csv
        deepJetXXXX.csv
    btagging_combination/
        deepCSVXXXX.csv
        deepCSV-yearCorrelation.csv
        ...
UL2017/
    btagging_ptRel/
        deepCSVXXXX.csv
        deepJetXXXX.csv
    subjet_tagging/
        deepCSVXXXX.csv
    ctagging_wcharm/
        deepCSVXXXX.csv
        deepJetXXXX.csv
    btagging_combination/
        deepCSVXXXX.csv
        deepCSV-yearCorrelation.csv
        ...
...
```
    
