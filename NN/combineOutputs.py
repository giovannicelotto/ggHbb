import numpy as np
import pandas as pd
import glob

path = "/t3home/gcelotto/ggHbb/scripts/NN/NNoutputFiles"
signalFileNames = glob.glob(path + "/y1*.parquet")
realDataFileNames = glob.glob(path + "/y0*.parquet")
signalOutput = pd.read_parquet(signalFileNames)
