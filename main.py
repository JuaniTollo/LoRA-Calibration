# main.py

from data_loader import get_folders_with_logits, load_npy_files
from model_calibrator import calibrate_model
from visualizer import scores_distribution
from utils import ensure_dir
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Run calibration, evaluation, and visualization for logits.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the root directory containing datasets and logits.",
    )
    args = parser.parse_args()
    root_dir = args.root_dir

    # Ensure necessary directories exist
    ensure_dir('models')
    ensure_dir('outputs')

    # Get dataset paths
    #socialiqa_routes, hellaswag_routes = get_folders_with_logits(root_dir)

    def process_dataset(routes, dataset_name, model_name):

        base_path, fine_tuned_path = get_folders_with_logits(root_dir, dataset_name, model_name)
        
        base_test_logits, base_test_targets = load_npy_files(base_path, split='test')
        ft_test_logits, ft_test_targets = load_npy_files(fine_tuned_path, split='test')

        base_calibrated_logits,base_logits_held_out, base_targets_held_out = calibrate_model(base_test_logits, base_test_targets, model_name)
        ft_calibrated_logits,ft_logits_held_out, ft_targets_held_out = calibrate_model(ft_test_logits, ft_test_targets, model_name)

        tokenizer_path = "./tokenizer/tokenizer.json"

        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Visualize distributions
        merged_df = scores_distribution(
            base_test_logits=base_logits_held_out,
            base_test_targets=base_targets_held_out,
            base_test_calibrated_logits=base_calibrated_logits,
            ft_logits_held_out=ft_logits_held_out,
            ft_targets_held_out=ft_targets_held_out,
            ft_calibrated_logits=ft_calibrated_logits,
            dataset=dataset_name,
            model_name=model_name,
            tokenizer=tokenizer
        )
        merged_df.to_csv(f"./outputs/{dataset_name}_{model_name}.csv")
    # Process datasets
    process_dataset(root_dir, "SocialIQA", "phi-1.5")
    process_dataset(root_dir, "SocialIQA",  "phi-2")
    process_dataset(root_dir, "HellaSwag",  "phi-1.5")
    process_dataset(root_dir, "HellaSwag",  "phi-2")

    return 
if __name__ == "__main__":
    main()
