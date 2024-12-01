def scores_distribution(
    base_logits,
    base_targets,
    fine_tuned_logits,
    fine_tuned_targets,
    calibrated_base_logits,
    calibrated_fine_tuned_logits,
    dataset,
    model_name
):
    import numpy as np
    import matplotlib.pyplot as plt
    from transformers import PreTrainedTokenizerFast

    # Load tokenizer and decode target labels
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tokenizer.json")
    decoded_base_targets = tokenizer.batch_decode(base_targets, skip_special_tokens=True)
    decoded_fine_tuned_targets = tokenizer.batch_decode(fine_tuned_targets, skip_special_tokens=True)
    decoded_original_targets = tokenizer.batch_decode(base_targets, skip_special_tokens=True)
    unique_targets = np.unique(
        decoded_base_targets + decoded_fine_tuned_targets + decoded_original_targets
    )

    # Create a mapping of target strings to numerical indices
    target_to_index = {target: idx for idx, target in enumerate(unique_targets)}

    # Ensure logits have correct dimensions
    def ensure_2d(logits):
        return logits if logits.ndim == 2 else np.expand_dims(logits, axis=1)

    base_logits = ensure_2d(base_logits)
    fine_tuned_logits = ensure_2d(fine_tuned_logits)
    calibrated_base_logits = ensure_2d(calibrated_base_logits)
    calibrated_fine_tuned_logits = ensure_2d(calibrated_fine_tuned_logits)

    # Softmax function
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    # Map logits to valid classes
    def map_logits_to_targets(logits, target_to_index):
        mapped_logits = np.zeros((logits.shape[0], len(target_to_index)))
        for i, target in enumerate(target_to_index):
            if i < logits.shape[1]:  # Ensure valid mapping
                mapped_logits[:, target_to_index[target]] = logits[:, i]
        return mapped_logits

    # Map logits to unique targets
    base_logits_mapped = map_logits_to_targets(base_logits, target_to_index)
    fine_tuned_logits_mapped = map_logits_to_targets(fine_tuned_logits, target_to_index)
    calibrated_base_logits_mapped = map_logits_to_targets(calibrated_base_logits, target_to_index)
    calibrated_fine_tuned_logits_mapped = map_logits_to_targets(calibrated_fine_tuned_logits, target_to_index)

    # Process logits
    base_scores = softmax(base_logits_mapped)
    fine_tuned_scores = softmax(fine_tuned_logits_mapped)
    calibrated_base_scores = softmax(calibrated_base_logits_mapped)
    calibrated_fine_tuned_scores = softmax(calibrated_fine_tuned_logits_mapped)

    # Calculate argmax distributions
    def calculate_distribution(argmax_labels, num_targets):
        counts = np.zeros(num_targets, dtype=int)
        for label in argmax_labels:
            counts[label] += 1
        return counts

    base_argmax_labels = np.argmax(base_scores, axis=1)
    fine_tuned_argmax_labels = np.argmax(fine_tuned_scores, axis=1)
    calibrated_base_argmax_labels = np.argmax(calibrated_base_scores, axis=1)
    calibrated_fine_tuned_argmax_labels = np.argmax(calibrated_fine_tuned_scores, axis=1)

    num_targets = len(unique_targets)
    base_distributions = calculate_distribution(base_argmax_labels, num_targets)
    fine_tuned_distributions = calculate_distribution(fine_tuned_argmax_labels, num_targets)
    calibrated_base_distributions = calculate_distribution(calibrated_base_argmax_labels, num_targets)
    calibrated_fine_tuned_distributions = calculate_distribution(calibrated_fine_tuned_argmax_labels, num_targets)

    # Original target distribution
    target_distributions = calculate_distribution(
        [target_to_index[label] for label in decoded_original_targets if label in target_to_index], num_targets
    )

    # Convert distributions to percentages
    total_samples = len(base_targets)
    base_percentages = 100 * np.array(base_distributions) / total_samples
    fine_tuned_percentages = 100 * np.array(fine_tuned_distributions) / total_samples
    calibrated_base_percentages = 100 * np.array(calibrated_base_distributions) / total_samples
    calibrated_fine_tuned_percentages = 100 * np.array(calibrated_fine_tuned_distributions) / total_samples
    target_percentages = 100 * np.array(target_distributions) / total_samples

    # Plotting
    indices = np.arange(len(unique_targets))
    width = 0.15  # Adjust width to fit all bars

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(indices - 2 * width, base_percentages, width, label='Base', color='lightblue')
    ax.bar(indices - width, fine_tuned_percentages, width, label='Fine-Tuned', color='lightgreen')
    ax.bar(indices, calibrated_base_percentages, width, label='Base Calibrated', color='skyblue')
    ax.bar(indices + width, calibrated_fine_tuned_percentages, width, label='Fine-Tuned Calibrated', color='salmon')
    ax.bar(indices + 2 * width, target_percentages, width, label='Original Target', color='orange')

    # Configure legend and labels
    ax.set_ylabel('Score Percentage')
    ax.set_xticks(indices)
    ax.set_xticklabels(unique_targets, rotation=45, ha="right")
    ax.legend(title="Distributions", loc="upper right")
    ax.set_title(f"{model_name} {dataset} - Label Distributions")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"./outputs/{model_name}_{dataset}_Score_Distributions.png", dpi=300)
    plt.close()
