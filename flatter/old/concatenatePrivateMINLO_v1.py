# %%
import pandas as pd
import glob

# Path to your parquet files
parquet_path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/GluGluH_M125_ToBB_private/*.parquet"

# Get list of all parquet files
parquet_files = glob.glob(parquet_path)

# Check if any files are found
if not parquet_files:
    raise FileNotFoundError(f"No files found at {parquet_path}")

# Read and concatenate all parquet files into one DataFrame
df_all = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
# %%
df_all.to_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/combined_private.parquet")
# Optionally print or save the result
print(f"Total rows: {len(df_all)}")
# df_all.to_parquet("concatenated_output.parquet")

# %%
