# %%
import pandas as pd
import dcor
# %%
path ="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df"
df = pd.read_parquet(path+"/df_dd_TTToHadronic_Jan14_900p0.parquet")

# %%

corr = dcor.distance_correlation(df.PNN1.values, df.PNN2.values)

# %%
