# To make this script work every 10 min:
# open crontab in the terminal by typing:
#   crontab -e
# Modify it in the following way:
# */10 * * * * ~/miniconda3/envs/myenv/bin/python /t3home/gcelotto/ggHbb/scripts/moveFiles.py

import os
import shutil
import glob
import numpy as np

def move_numpy_files(source_directory, destination_directory, min_size_mb=1):
    # Ensure that the source and destination directories exist
    if not os.path.exists(source_directory):
        print(f"Source directory '{source_directory}' does not exist.")
        return

    if not os.path.exists(destination_directory):
        print(f"Destination directory '{destination_directory}' does not exist.")
        return

    # Use glob to find all numpy files in the source directory
    numpy_files = glob.glob(os.path.join(source_directory, '*.npy'))
    print(len(numpy_files))
    min_size_bytes = min_size_mb * 1024  * 1024 # Convert min_size_mb to bytes

    print("List of numpy files found in the source folder:")
    filtered_numpy_files = []
    for file_path in numpy_files:
        file_size = os.path.getsize(file_path)
        if file_size > min_size_bytes:
            filtered_numpy_files.append(file_path)
            print(file_path)
        else:
            pass
    
    print(len(filtered_numpy_files))
    if (not filtered_numpy_files) | (len(filtered_numpy_files)==0):
        print(f"No numpy files found in '{source_directory}'.")
        return
    #while True:
    #    user_input = input("Continue? (y/n): ").lower()
    #    if user_input == 'y':
    #        break
    #    elif user_input == 'n':
    #        print("Stopped.")
    #        return
    #    else:
    #        print("Invalid input. Please enter 'y' or 'n'.")

    # Move each numpy file to the destination directory if its size is greater than min_size_mb
    for numpy_file in filtered_numpy_files:
        file_name = os.path.basename(numpy_file)
        destination_path = os.path.join(destination_directory, file_name)

        # Check if the file already exists in the destination directory
        if os.path.exists(destination_path):
            print(f"File '{file_name}' already exists in the destination directory.")
        else:
            # Check the size of the file (in bytes)
            file_size = os.path.getsize(numpy_file)
            min_size_bytes = min_size_mb * 1024 * 1024  # Convert min_size_mb to bytes

            if file_size > min_size_bytes:
                try:
                    currentFile = np.load(numpy_file)
                    shutil.move(numpy_file, destination_path)
                    print(f"Moved '{file_name}' to '{destination_directory}'.")
                except:
                    print(f"Skipped '{file_name}' due to errors in the attempt to open it.")

            else:
                print(f"Skipped '{file_name}' due to size less than {min_size_mb} MB.")

# Example usage
source_directorySignal = '/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/ggHTrue'
destination_directorySignal = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/flatData'

source_directoryBkg = '/t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/data'
destination_directoryBkg = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/flatData'

#signal = False
#source_directory = source_directorySignal if signal else source_directoryBkg
#destination_directory = destination_directorySignal if signal else destination_directoryBkg
min_size_mb = 1
print("SIGNAL")
move_numpy_files(source_directorySignal, destination_directorySignal, min_size_mb)
print("\nBACKGROUND")
move_numpy_files(source_directoryBkg, destination_directoryBkg, min_size_mb)


# Signal
# /t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/data
# /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures

# Data
# /t3home/gcelotto/bbar_analysis/flatData/selectedCandidates/ggHTrue
# /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures

