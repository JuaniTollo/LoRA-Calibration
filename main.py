# main.py

from data_loader import get_folders_with_logits, load_npy_files, divide_test_and_cal
from model_calibrator import calibrate_and_evaluate
from visualizer import scores_distribution
from utils import ensure_dir
import pandas as pd

def main():
    # Ensure necessary directories exist
    ensure_dir('models')
    ensure_dir('outputs')

    # Replace with your actual root directory
    root_dir = ".../experimentos/poutput"
    socialiqa_routes, hellaswag_routes = get_folders_with_logits(root_dir)
    df = pd.DataFrame(columns=["model", "overall_perf", "overall_perf_after_cal", "cal_loss", "rel_cal_loss"])

    for dataset_routes in [socialiqa_routes, hellaswag_routes]:
        for route in dataset_routes:
            print(f"Processing route: {route}")
            test_logits, test_targets = load_npy_files(route, split='test')
            scores_trn, targets_trn, scores_tst, targets_tst = divide_test_and_cal(test_logits, test_targets)
            model_name = route.replace("/", "_")
            results = calibrate_and_evaluate(scores_tst, scores_trn, targets_trn, targets_tst, model_name)
            df.loc[len(df)] = [route, *results.values()]

    # Save results to CSV
    df.to_csv('models/calibration_results.csv', index=False)
    print("Calibration results saved to 'models/calibration_results.csv'.")

    # Visualization Example
    # Adjust the paths and model names as necessary
    base_logits, base_targets = load_npy_files('path_to_base_model', split='val')
    fine_tuned_logits, fine_tuned_targets = load_npy_files('path_to_fine_tuned_model', split='val')
    scores_distribution(base_logits, base_targets, fine_tuned_logits, fine_tuned_targets, 'dataset_name', 'model_name')

if __name__ == "__main__":
    main()
