from mk_csv import CSVFile
import numpy as np

# create new csv file
# the arguments handed to the class initialization determine the output file name
csv = CSVFile(tagger="particleNet", descriptor="kinfit", version=0)

for _ in range(20):
    # add one new line to the csv file
    #   required arguments: 
    #       type, formula      
    #   optional arguments (with defaults):
    #       syst="central", 
    #       wp="-", 
    #       flav=5,
    #       etaMin=0, etaMax=2.5,
    #       ptMin=20, ptMax=1000,
    #       discrMin=0, discrMax=1
    csv.add_line(type="kinfit",
        wp="L",
        syst="central",
        ptMin=20, ptMax=30,
        formula=np.random.random()
        )
 
# print csv file in pandas dataframe format       
print(csv)

# write csv file to path
csv.write(path=".")
