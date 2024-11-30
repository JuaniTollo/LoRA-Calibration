# main.py

from data_loader import get_folders_with_logits, load_npy_files, divide_test_and_cal
from model_calibrator import calibrate_and_evaluate
from visualizer import scores_distribution
from utils import ensure_dir, ensure_tokenizer
import pandas as pd
import argparse
import os 

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run calibration and evaluation for logits.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the root directory containing the datasets and logits.",
    )
    args = parser.parse_args()
    root_dir = args.root_dir

    # Ensure necessary directories exist
    ensure_dir('models')
    ensure_dir('outputs')

    # Load dataset routes
    socialiqa_routes, hellaswag_routes = get_folders_with_logits(root_dir)
    # df = pd.DataFrame(columns=["model", "overall_perf", "overall_perf_after_cal", "cal_loss", "rel_cal_loss"])

    # for dataset_routes in [socialiqa_routes, hellaswag_routes]:
    #     for route in dataset_routes:
    #         print(f"Processing route: {route}")
    #         test_logits, test_targets = load_npy_files(route, split='test')
    #         scores_trn, targets_trn, scores_tst, targets_tst = divide_test_and_cal(test_logits, test_targets)
    #         model_name = route.replace("/", "_")
    #         results = calibrate_and_evaluate(scores_tst, scores_trn, targets_trn, targets_tst, model_name)
    #         df.loc[len(df)] = [route, *results.values()]

    # # Save results to CSV
    # df.to_csv('models/calibration_results.csv', index=False)
    # print("Calibration results saved to 'models/calibration_results.csv'.")

    ensure_tokenizer(tokenizer_path="./tokenizer/tokenizer.json")

    # Define datasets and model versions to iterate over
    datasets = ["hellaswag", "socialiqa"]
    versions = ["phi-1.5", "phi-2"]

    for version in versions:
        for dataset in datasets:
            base_model_dir = os.path.join(root_dir, "microsoft", version, dataset, "Base")
            fine_tuned_model_dir = os.path.join(root_dir, "microsoft", version, dataset)

            # Handle base models
            if os.path.exists(base_model_dir):
                base_logits, base_targets = load_npy_files(base_model_dir, split="val")

                # Find fine-tuned models by navigating subdirectories
                for root, dirs, files in os.walk(fine_tuned_model_dir):
                    if "Base" not in root and "test_all_logits.npy" in files:
                        fine_tuned_logits, fine_tuned_targets = load_npy_files(root, split="val")
                        
                        # Extract the fine-tuned subdirectory as the model name
                        model_name = root.replace(root_dir, "").strip("/").replace("/", "_")
                        
                        print(f"Processing: Base Model vs Fine-Tuned Model '{model_name}'")
                        scores_distribution(
                            base_logits,
                            base_targets,
                            fine_tuned_logits,
                            fine_tuned_targets,
                            dataset=dataset,
                            model_name=model_name,
                        )
if __name__ == "__main__":
    main()

#     root_dir = "/Users/juantollo/Library/Mobile Documents/com~apple~CloudDocs/Tesis/experimentos/output copy"
