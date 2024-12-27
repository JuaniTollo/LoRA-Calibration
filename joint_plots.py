import matplotlib.pyplot as plt
import pandas as pd
import os
def calibration_plots2(folder_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    i = 0

    for csv_file in os.listdir(folder_dir):
        if "performance" in csv_file:
            dataset_name = csv_file.split("_")[1].split(".")[0]
            model_name = csv_file.split("_")[2].split(".")[0]

            df = pd.read_csv(os.path.join(folder_dir, csv_file))
            ax = axes[i]

            base = df[df["base/fine tuned"] == "base"]
            fine_tuned = df[df["base/fine tuned"] == "fine-tuned"]

            base_values = [base["overall_perf"].values[0], base["overall_perf_after_cal"].values[0]]
            fine_tuned_values = [fine_tuned["overall_perf"].values[0], fine_tuned["overall_perf_after_cal"].values[0]]
            base_rel_cal_loss = base["rel_cal_loss"].values[0]
            fine_tuned_rel_cal_loss = fine_tuned["rel_cal_loss"].values[0]

            ax.bar(0, base_values[0], color='blue', width=0.8, label="Before calibration")
            ax.bar(1, base_values[1], color='lightblue', width=0.8, label="After calibration")
            ax.text(0.5, base_values[0] + 0.02, f"{base_rel_cal_loss:.2f}", ha='center', fontsize=12)

            ax.bar(2.5, fine_tuned_values[0], color='blue', width=0.8)
            ax.bar(3.5, fine_tuned_values[1], color='lightblue', width=0.8)
            ax.text(3, fine_tuned_values[0] + 0.02, f"{fine_tuned_rel_cal_loss:.2f}", ha='center', fontsize=12)

            ax.set_title(model_name.capitalize() + " " + dataset_name)
            ax.set_xticks([0.5, 3.])
            ax.set_xticklabels(["Base", "Fine-tuned"])
            ax.set_ylabel("Performance")
            ax.grid(axis='y', alpha=0.7)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.legend(loc="upper right")

            i += 1

    plt.tight_layout()
    plt.savefig("plots/calibration_with_rel_loss.png")
    plt.show()

if __name__ == "__main__":
    calibration_plots2("outputs/")
