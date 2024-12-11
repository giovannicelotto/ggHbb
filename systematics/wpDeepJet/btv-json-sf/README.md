# btv-json-sf

Scripts to convert and test b-tagging SFs from BTV POG in json format and to homogenize some information of the csv files

## correctionlib

To run the scripts provided here the correctionlib v2 is needed.
See the [documentation](https://cms-nanoaod.github.io/correctionlib/schemav2.html) and the [repository](https://github.com/cms-nanoAOD/correctionlib) with instructions how to set it up correctly.


## Instructions

To convert csv files to json files the input csv files are needed.  
In an effort to centralize the collection of available SFs for different methods and taggers a repository is available [here](https://gitlab.cern.ch/cms-btv/btv-scale-factors).

Until this repository is filled with all the information an `input` directory exists here storing the csv files that are used currently.

New json files can be created with the `produce_X` scripts. As an example, this can be used to create AK4 b-tagging SF json and csv files:
```bash
python3 produce_btagging_json.py -y 2018
```

After the json files have been created a summary of its contents can be printed to the shell via 
```bash
python3 -m correctionlib.cli summary FILE
```

Control plots per tagger, per year, per working point, etc can be created with the `plot_X` scripts, e.g. via
```bash
python3 plot_fixedWP_sf.py -y 2018 --wp "T" --tagger deepJet -v 1
```

Furthermore a test script is available to test the AK4 b-tagging SFs, e.g. via
```bash
python3 test_sf_json.py -y 2018
```

The json files and updated csv files are stored in the `data/` directory and are ready to be used.

To be provided to the CMS collaboration the json files need to be gzipped
```bash
gzip FILE
```
and added to the [jsonPOG repository](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration) via merge request.
Additionally, the csv files are provided in the different BTV TWiki pages.

