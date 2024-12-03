# data_loader.py

import numpy as np
import os

def categorize_routes(root, phi_version, dataset_name, base_routes, fine_tuned_routes):
    """
    Categorize routes into base and fine-tuned lists based on the phi version and dataset name.

    Args:
        root (str): The directory path to categorize.
        phi_version (str): The phi version to filter (e.g., "phi-1.5", "phi-2").
        dataset_name (str): The dataset name to filter (e.g., "socialiqa", "hellaswag").
        base_routes (list): List to append base routes.
        fine_tuned_routes (list): List to append fine-tuned routes.
    """
    if dataset_name in root.lower() and phi_version in root.lower():
        if "base" in root.lower():
            base_routes.append(root)
        else:
            fine_tuned_routes.append(root)

import os

def get_folders_with_logits(root_dir, dataset_name, phi_version):
    """
    Get paths for a specific dataset and phi model, categorized into base and fine-tuned routes.

    Args:
        root_dir (str): The root directory containing the datasets.
        dataset_name (str): The dataset name to filter (e.g., "socialiqa", "hellaswag").
        phi_version (str): The phi version to filter (e.g., "phi-1.5", "phi-2").

    Returns:
        tuple: A tuple containing two lists:
            - Base model routes
            - Fine-tuned model routes
    """
    base_routes = []
    fine_tuned_routes = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy') and "adapters" not in root:
                if dataset_name.lower() in root.lower() and phi_version.lower() in root.lower():
                    if "base" in root.lower():
                        base_routes.append(root)
                    else:
                        fine_tuned_routes.append(root)

    return list(set(base_routes))[0], list(set(fine_tuned_routes))[0]


def load_npy_files(model_path, split='test'):
    logits = np.load(f'{model_path}/{split}_all_logits.npy')
    targets = np.load(f'{model_path}/{split}_all_targets.npy')
    return logits, targets
