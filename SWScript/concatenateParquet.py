# %%
import pandas as pd
import glob
source = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB_private/"
target = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/GluGluHToBBMINLO_tr_999.parquet"
fileNames = glob.glob(source + "/*.parquet")
# %%
df = pd.read_parquet(fileNames)
# %%
df.to_parquet(target)
# %%
if df.isna().sum().sum()>0:
    print("There are %d Nan. Press to drop the nan and save the filtered DataFrame"%df.isna().sum().sum())
    df.isna().sum()
    df = df.dropna()
    df.to_parquet(target)