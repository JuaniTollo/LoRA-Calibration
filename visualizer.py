import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def scores_distribution(
    base_test_logits,
    base_test_targets,
    base_test_calibrated_logits,
    ft_logits_held_out,
    ft_targets_held_out,
    ft_calibrated_logits,
    dataset,
    model_name,
    tokenizer=None
):
    """
    Visualize and compute distributions of logits and targets.

    Args:
        base_test_logits (ndarray): Logits for base model.
        base_test_targets (ndarray): Targets for base model.
        base_test_calibrated_logits (ndarray): Calibrated logits for base model.
        ft_logits_held_out (ndarray): Logits for fine-tuned model.
        ft_targets_held_out (ndarray): Targets for fine-tuned model.
        ft_calibrated_logits (ndarray): Calibrated logits for fine-tuned model.
        dataset (str): Dataset name.
        model_name (str): Model name.
        tokenizer (PreTrainedTokenizer, optional): Tokenizer to decode target indices.
    """

    # Helper: softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    # Helper: calculate counts
    def calculate_distribution(scores, targets):
        argmax_labels = np.argmax(scores, axis=1)
        unique_targets = np.unique(targets)
        counts_dict = {target: 0 for target in unique_targets}
        for label in argmax_labels:
            if label in counts_dict:
                counts_dict[label] += 1
        df = pd.DataFrame(list(counts_dict.items()), columns=["Target_Index", "Count"])
        if tokenizer:
            df["Vocabulary"] = df["Target_Index"].apply(lambda x: tokenizer.decode([x]))
        return df

    # Prepare scores
    base_scores = softmax(base_test_logits)
    base_calibrated_scores = softmax(base_test_calibrated_logits)
    ft_scores = softmax(ft_logits_held_out)
    ft_calibrated_scores = softmax(ft_calibrated_logits)

    # Calculate distributions
    base_distribution = calculate_distribution(base_scores, base_test_targets)
    base_calibrated_distribution = calculate_distribution(base_calibrated_scores, base_test_targets)
    ft_distribution = calculate_distribution(ft_scores, ft_targets_held_out)
    ft_calibrated_distribution = calculate_distribution(ft_calibrated_scores, ft_targets_held_out)

    # Merge datasets for unified view
    merged = pd.concat([
        base_distribution.rename(columns={"Count": "Base"}),
        base_calibrated_distribution.rename(columns={"Count": "Base_Calibrated"}),
        ft_distribution.rename(columns={"Count": "Fine_Tuned"}),
        ft_calibrated_distribution.rename(columns={"Count": "Fine_Tuned_Calibrated"})
    ], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].fillna(0)

    # Calculate and map counts for the `Target` column
    decoded_targets = tokenizer.batch_decode(base_test_targets, skip_special_tokens=True)
    counts_dict = Counter(decoded_targets)
    merged["Target"] = merged["Vocabulary"].map(counts_dict).fillna(0).astype(int)

    # Normalize columns
    columns_to_normalize = ["Base", "Base_Calibrated", "Fine_Tuned", "Fine_Tuned_Calibrated", "Target"]
    merged[columns_to_normalize] = merged[columns_to_normalize].div(merged[columns_to_normalize].sum(axis=0), axis=1)

    # Plot distributions
    melted_df = pd.melt(
        merged,
        id_vars=["Vocabulary"],
        value_vars=columns_to_normalize,
        var_name="Metric",
        value_name="Value"
    )
    merged_sorted = merged.sort_values(by="Vocabulary", ascending=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Vocabulary", y="Value", hue="Metric", data=melted_df, palette="viridis")

    plt.xlabel("Vocabulary", fontsize=14)
    plt.ylabel("Proportion", fontsize=14)
    plt.title(f"Labels distribution run on {model_name} for {dataset}", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Metric", fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}_{model_name}.png", dpi=300)
    return merged

import pandas as pd
import seaborn as sns

import seaborn as sns

def create_performance_plot(dataframe, save_path, model_name, dataset_name):
    """
    Generates a performance plot for the given DataFrame showing before and after calibration.
    Adds delta (rel_cal_loss) as text annotations and includes dataset_name and model_name in the plot title.

    :param dataframe: Input pandas DataFrame
    :param save_path: Output path to save the plot
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    """
    # Melt the DataFrame for seaborn compatibility
    melted_df = dataframe.melt(
        id_vars=["model", "base/fine tuned", "dataset_name", "rel_cal_loss"],
        value_vars=["overall_perf", "overall_perf_after_cal"],
        var_name="Calibration",
        value_name="Performance"
    )
    
    # Create barplot with Seaborn
    sns.set_theme(style="whitegrid")
    plot = sns.catplot(
        data=melted_df,
        x="base/fine tuned",
        y="Performance",
        hue="Calibration",
        kind="bar",
        height=6,
        aspect=1.5,
        palette="muted"
    )
    
    # Add delta (rel_cal_loss) as annotations
    for i, row in dataframe.iterrows():
        calibrated_value = row["overall_perf_after_cal"]
        rel_cal_loss = row["rel_cal_loss"]
        base_fine_tuned = row["base/fine tuned"]
        
        # Find the corresponding bar for annotation
        bar_index = melted_df[
            (melted_df["Calibration"] == "overall_perf_after_cal") &
            (melted_df["base/fine tuned"] == base_fine_tuned)
        ].index[0]
        bar = plot.ax.patches[bar_index]
        
        # Add annotation
        plot.ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"Δ {rel_cal_loss:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )
    
    # Set titles and labels
    plot.set_axis_labels("Model Type", "Performance")
    plot.fig.suptitle(
        f"Performance Before and After Calibration\nDataset: {dataset_name}, Model: {model_name}",
        y=1.02
    )

    # Save the plot
    plot.savefig(save_path)
    sns.reset_defaults()
