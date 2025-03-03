# Python script to submit a gridSearch
# Batch size
# Learning rate
# Lambda_dcor
# nNodes
# activation function (?) to be implemented
import subprocess
import random

# Define the hyperparameter grid
lambda_disco_values = ["800", "900", "1000"]  # Example values
bs_values = ["8192", "16384", "32768"]
lr_values = ["1e-4", "5e-4", "1e-3"]
nNodes_values = ["128,64,32", "64,64,64", "256,256"]  # Ensure format is correct
model_base_name = "DD"

# Loop over all combinations
for i, lambda_disco in enumerate(lambda_disco_values):
    for j, bs in enumerate(bs_values):
        for k, lr in enumerate(lr_values):
            for l, nNodes in enumerate(nNodes_values):
                modelName = f"{model_base_name}_{i}{j}{k}{l}"  # Unique model name
                print(f"Submitting job: λ={lambda_disco}, bs={bs}, lr={lr}, nNodes={nNodes}, model={modelName}")

                subprocess.run([
                    'sbatch', '-J', modelName,
                    '/t3home/gcelotto/ggHbb/PNN/job.sh',
                    lambda_disco, bs, lr, nNodes, modelName
                ])
