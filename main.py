from data_loader import get_folders_with_logits, load_npy_files, divide_test_and_cal
from model_calibrator import calibrate_and_evaluate
from visualizer import scores_distribution
from utils import ensure_dir, ensure_tokenizer
import pandas as pd
import argparse
import os
import numpy as np

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
    df = pd.DataFrame(columns=["model", "overall_perf", "overall_perf_after_cal", "cal_loss", "rel_cal_loss"])

    # Helper function to aggregate logits and targets
    def aggregate_logits_and_targets(routes):
        aggregated_logits = []
        aggregated_targets = []
        calibrated_base_logits = []
        calibrated_fine_tuned_logits = []

        for route in routes:
            print(f"Processing route: {route}")
            test_logits, test_targets = load_npy_files(route, split='test')
            scores_trn, targets_trn, scores_tst, targets_tst = divide_test_and_cal(test_logits, test_targets)
            model_name = route.replace("/", "_")

            results = calibrate_and_evaluate(scores_tst, scores_trn, targets_trn, targets_tst, model_name)
            df.loc[len(df)] = [
                route,
                results["overall_perf"],
                results["overall_perf_after_cal"],
                results["cal_loss"],
                results["rel_cal_loss"],
            ]

            # Collect logits and targets for aggregation
            aggregated_logits.append(test_logits)
            aggregated_targets.append(test_targets)
            calibrated_base_logits.append(results["calibrated_logits_base"])
            calibrated_fine_tuned_logits.append(results["calibrated_logits_fine_tuned"])

        # Concatenate logits and targets for plotting
        return (
            np.concatenate(aggregated_logits, axis=0),
            np.concatenate(aggregated_targets, axis=0),
            np.concatenate(calibrated_base_logits, axis=0),
            np.concatenate(calibrated_fine_tuned_logits, axis=0),
        )

    # Aggregate data for SocialIQA
    socialiqa_logits, socialiqa_targets, socialiqa_calibrated_base, socialiqa_calibrated_fine_tuned = aggregate_logits_and_targets(
        socialiqa_routes
    )

    # Aggregate data for HellaSwag
    hellaswag_logits, hellaswag_targets, hellaswag_calibrated_base, hellaswag_calibrated_fine_tuned = aggregate_logits_and_targets(
        hellaswag_routes
    )

    # Generate plots for SocialIQA
    scores_distribution(
        base_logits=socialiqa_logits,
        base_targets=socialiqa_targets,
        fine_tuned_logits=socialiqa_logits,  # Adjust if separate fine-tuned logits exist
        fine_tuned_targets=socialiqa_targets,  # Adjust if separate fine-tuned targets exist
        calibrated_base_logits=socialiqa_calibrated_base,
        calibrated_fine_tuned_logits=socialiqa_calibrated_fine_tuned,
        dataset="SocialIQA",
        model_name="phi-1.5",
    )

    # Generate plots for HellaSwag
    scores_distribution(
        base_logits=hellaswag_logits,
        base_targets=hellaswag_targets,
        fine_tuned_logits=hellaswag_logits,  # Adjust if separate fine-tuned logits exist
        fine_tuned_targets=hellaswag_targets,  # Adjust if separate fine-tuned targets exist
        calibrated_base_logits=hellaswag_calibrated_base,
        calibrated_fine_tuned_logits=hellaswag_calibrated_fine_tuned,
        dataset="HellaSwag",
        model_name="phi-1.5",
    )

    # Save results to CSV
    df.to_csv('models/calibration_results.csv', index=False)
    print("Calibration results saved to 'models/calibration_results.csv'.")

if __name__ == "__main__":
    main()
