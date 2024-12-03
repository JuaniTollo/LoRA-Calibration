# main.py

from data_loader import get_folders_with_logits, load_npy_files
from model_calibrator import calculate_logloss_and_calibrate
from visualizer import scores_distribution, create_performance_plot
from utils import ensure_dir
from transformers import PreTrainedTokenizerFast
import argparse
import numpy as np
import os
import pandas as pd

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
        df = pd.DataFrame(columns=["model", "base/fine tuned","dataset_name","overall_perf", "overall_perf_after_cal", "cal_loss", "rel_cal_loss"])

        base_path, fine_tuned_path = get_folders_with_logits(root_dir, dataset_name, model_name)
        
        base_test_logits, base_test_targets = load_npy_files(base_path, split='test')
        ft_test_logits, ft_test_targets = load_npy_files(fine_tuned_path, split='test')

        dict_with_answers_base, base_scores_tst, base_targets_tst, base_scores_tst_cal = calculate_logloss_and_calibrate(base_test_logits, base_test_targets, model_name)
        df.loc[len(df)] = [f"{model_name}", f"base", f"{dataset_name}", *dict_with_answers_base.values()]

        dict_with_answers_ft, ft_scores_tst, ft_targets_tst, ft_scores_tst_cal = calculate_logloss_and_calibrate(ft_test_logits, ft_test_targets, model_name)
        df.loc[len(df)] = [f"{model_name}", "fine-tuned", f"{dataset_name}", *dict_with_answers_ft.values()]

        df.to_csv(f"./outputs/performance_{dataset_name}_{model_name}.csv")
        
        tokenizer = load_tokenizer()
        
        # Visualize distributions
        merged_df = scores_distribution(
            base_test_logits=base_scores_tst,
            base_test_targets=base_targets_tst,
            base_test_calibrated_logits=base_scores_tst_cal,
            ft_logits_held_out=ft_scores_tst,
            ft_targets_held_out=ft_targets_tst,
            ft_calibrated_logits=ft_scores_tst_cal,
            dataset=dataset_name,
            model_name=model_name,
            tokenizer=tokenizer
        )
        csv_path = f"./outputs/distribution_{dataset_name}_{model_name}.csv"
        merged_df.to_csv(csv_path)
        
        df = pd.read_csv(f"./outputs/performance_{dataset_name}_{model_name}.csv")
        print(df)
        create_performance_plot(df, f"./plots/{dataset_name}_{model_name}.jpg", model_name, dataset_name)

    def load_tokenizer():
        tokenizer_path = "./tokenizer/tokenizer.json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        return tokenizer
    # Process datasets
    process_dataset(root_dir, "SocialIQA", "phi-1.5")
    process_dataset(root_dir, "SocialIQA",  "phi-2")
    process_dataset(root_dir, "HellaSwag",  "phi-1.5")
    process_dataset(root_dir, "HellaSwag",  "phi-2")

if __name__ == "__main__":
    main()
