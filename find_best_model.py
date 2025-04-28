"""
Loops through models and generates json with validation loss curves for each model

Samuel Gagnon-Hartman 2024
"""

import json
import os

model_dir = f'/leonardo_scratch/large/userexternal/sgagnonh/hyperparams_run/'
all_models = os.listdir(model_dir)
all_model_IDs = [int(x.split('TASKID')[1].split('_')[0]) for x in all_models]

loss_dict = {}
for model_ID in all_model_IDs:
    infile = f'{model_dir}TASKID{model_ID}/stdout_1.txt'
    val_loss = []
    with open(infile) as f:
        f = f.readlines()
    for line in f:
        if "epoch" in line:
            val_loss.append(float(line.split(" ")[-1]))
    loss_dict[str(model_ID)] = val_loss

# Convert and write JSON object to file
with open('./val_losses.json', 'w') as outfile: 
    json.dump(loss_dict, outfile)