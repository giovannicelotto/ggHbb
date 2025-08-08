import uproot
import awkward as ak
import numpy as np

def load_genZ_pt_from_folder(folder_path, branches=None):
    """
    Load and concatenate GenPart branches from all ROOT files in the folder,
    apply Z boson mask (statusFlags bit 13 and pdgId==23),
    and return flattened GenPart_pt for Z bosons.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing ROOT files.
    branches : list of str, optional
        List of branches to load. Default to necessary branches.
        
    Returns:
    --------
    ak.Array
        Flattened GenPart_pt array for Z bosons.
    """
    import glob
    
    if branches is None:
        branches = ["GenPart_pdgId", "GenPart_statusFlags", "GenPart_pt"]

    fileNames = glob.glob(folder_path + "/**/*.root", recursive=True)
    if not fileNames:
        raise RuntimeError(f"No ROOT files found in {folder_path}")
    
    # Concatenate branches from Events tree of all files
    arrays = uproot.concatenate(
        [f + ":Events" for f in fileNames],
        expressions=branches,
        library="ak",
    )

    GenPart_pdgId = arrays["GenPart_pdgId"]
    GenPart_statusFlags = arrays["GenPart_statusFlags"]
    GenPart_pt = arrays["GenPart_pt"]

    # Z boson mask: bit 13 of statusFlags and pdgId==23
    maskZ = ((GenPart_statusFlags & 8192) != 0) & (GenPart_pdgId == 23)

    # Check exactly one Z per event (optional, can comment out)
    oneZperEvent = ak.all(ak.sum(maskZ, axis=1) == 1)
    if not oneZperEvent:
        print(f"Warning: Not every event has exactly one Z boson in {folder_path}")

    return ak.flatten(GenPart_pt[maskZ])
