import glob
import os
import pandas as pd

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB_private_v2"
out_dir = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/training/v2"
os.makedirs(out_dir, exist_ok=True)

file_names = sorted(glob.glob(path + "/*.parquet"))
print(f"Found {len(file_names)} files")

chunk_size = 1000

for i in range(0, len(file_names), chunk_size):
    chunk_files = file_names[i:i + chunk_size]
    chunk_idx = i // chunk_size

    print(f"Merging files {i} → {i + len(chunk_files) - 1} into chunk {chunk_idx}")

    dfs = []
    for f in chunk_files:
        dfs.append(pd.read_parquet(f))

    df_merged = pd.concat(dfs, ignore_index=True)

    out_file = os.path.join(out_dir, f"merged_{chunk_idx:02d}.parquet")
    df_merged.to_parquet(out_file)

    # free memory explicitly
    del dfs, df_merged