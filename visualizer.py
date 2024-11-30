# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def scores_distribution(base_logits, base_targets, fine_tuned_logits, fine_tuned_targets, dataset, model_name):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tokenizer.json")
    decoded_base_targets = tokenizer.batch_decode(base_targets, skip_special_tokens=True)
    decoded_fine_tuned_targets = tokenizer.batch_decode(fine_tuned_targets, skip_special_tokens=True)
    unique_targets = np.unique(decoded_base_targets)

    base_scores = softmax(base_logits)
    fine_tuned_scores = softmax(fine_tuned_logits)
    base_scores_max = np.argmax(base_scores, axis=1)
    fine_tuned_scores_max = np.argmax(fine_tuned_scores, axis=1)

    base_target_scores = [np.sum(base_scores_max == i) for i in range(len(unique_targets))]
    fine_tuned_target_scores = [np.sum(fine_tuned_scores_max == i) for i in range(len(unique_targets))]

    base_percentages = 100 * np.array(base_target_scores) / base_scores_max.size
    fine_tuned_percentages = 100 * np.array(fine_tuned_target_scores) / fine_tuned_scores_max.size

    indices = np.arange(len(unique_targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(indices, base_percentages, width, label=f'{model_name} Base', color='lightblue')
    ax.bar(indices + width, fine_tuned_percentages, width, label=f'{model_name} Fine-Tuned', color='lightgreen')

    ax.set_ylabel('Score Percentage')
    ax.set_xticks(indices + width / 2)
    ax.set_xticklabels(unique_targets, rotation=45)
    ax.legend()
    ax.set_title(f"{model_name} {dataset}")

    plt.tight_layout()
    plt.savefig(f"./outputs/{model_name}_{dataset}_Score_Distributions.png")
    plt.show()
