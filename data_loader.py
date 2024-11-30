# data_loader.py

import numpy as np
import os
from sklearn.model_selection import train_test_split

def get_folders_with_logits(carpeta):
    hellaswag_routes = []
    socialiqa_routes = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.npy') and "adapters" not in root:
                if "socialiqa" in root:
                    socialiqa_routes.append(root)
                elif "hellaswag" in root:
                    hellaswag_routes.append(root)
    return list(set(socialiqa_routes)), list(set(hellaswag_routes))

def load_npy_files(model_path, split='test'):
    logits = np.load(f'{model_path}/{split}_all_logits.npy')
    targets = np.load(f'{model_path}/{split}_all_targets.npy')
    return logits, targets

def divide_test_and_cal(logits, targets, prop=0.2):
    indices_calibrator, indices_held_out = train_test_split(
        range(len(targets)), test_size=1 - prop, stratify=targets
    )
    logits_calibrator = logits[indices_calibrator]
    targets_calibrator = targets[indices_calibrator]
    logits_held_out = logits[indices_held_out]
    targets_held_out = targets[indices_held_out]
    return logits_calibrator, targets_calibrator, logits_held_out, targets_held_out
