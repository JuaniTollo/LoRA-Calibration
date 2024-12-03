# model_calibrator.py

import numpy as np
import torch
from expected_cost.calibration import calibration_train_on_heldout
from expected_cost.psrcal_wrappers import LogLoss
from psrcal.calibration import AffineCalLogLoss
from sklearn.model_selection import train_test_split
import scipy

def divide_test_and_cal(logits, targets, prop=0.2):
    """Split logits and targets into calibration and held-out subsets."""
    indices_cal, indices_held_out = train_test_split(
        range(len(targets)), test_size=1 - prop, stratify=targets
    )
    return (
        logits[indices_cal], targets[indices_cal],
        logits[indices_held_out], targets[indices_held_out]
    )

def calculate_log_loss(logits, targets):
    """Compute log loss."""
    log_softmax_vals = scipy.special.log_softmax(logits, axis=1)
    return LogLoss(log_softmax_vals, targets, priors=None, norm=True)

def calibrate_model(full_logits, full_targets, model_name, calibration_prop=0.2):
    """
    Calibrate model on a subset and save calibrated results for held-out set.

    Args:
        full_logits (np.ndarray): Logits to calibrate.
        full_targets (np.ndarray): Corresponding targets.
        model_name (str): Model name for saving calibration model.
        calibration_prop (float): Proportion of data used for calibration.

    Returns:
        dict: Calibrated and held-out logits and metrics.
    """
    # Divide data into calibration and held-out sets
    logits_cal, targets_cal, logits_held_out, targets_held_out = divide_test_and_cal(
        full_logits, full_targets, prop=calibration_prop
    )

    # Calibrate using the calibration subset
    calibrated_logits, calibration_model = calibration_train_on_heldout(
        logits_held_out, logits_cal, targets_cal,
        calparams={'bias': True, 'priors': None},
        calmethod=AffineCalLogLoss,
        return_model=True
    )
    torch.save(calibration_model.state_dict(), f'models/{model_name}_calibration.pth')

    return calibrated_logits,logits_held_out, targets_held_out
