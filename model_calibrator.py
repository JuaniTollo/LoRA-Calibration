# model_calibrator.py

import numpy as np
import torch
from expected_cost.calibration import calibration_train_on_heldout
from expected_cost.psrcal_wrappers import LogLoss
from psrcal.calibration import AffineCalLogLoss
from sklearn.model_selection import train_test_split
import scipy
def calculate_accuracy(logits, targets):
    """
    Calcula la accuracy dado un array de logits y un vector de targets.
    
    Args:
        logits (np.ndarray): Matriz de logits de forma (n_clases, n_muestras).
        targets (np.ndarray): Vector de targets de largo n_muestras.
        
    Returns:
        float: Porcentaje de precisión (accuracy).
    """
    # Obtener las predicciones seleccionando el índice máximo (clase más probable)
    predictions = np.argmax(logits, axis=0)
    
    # Comparar predicciones con los targets
    correct_predictions = np.sum(predictions == targets)
    
    # Calcular precisión
    accuracy = correct_predictions / len(targets)
    return accuracy

def divide_test_and_cal(logits, targets, prop=0.2):
    """Split logits and targets into calibration and held-out subsets."""
    indices_cal, indices_held_out = train_test_split(
        range(len(targets)), test_size=1 - prop, stratify=targets
    )
    
    return (
        logits[indices_cal,:], targets[indices_cal],
        logits[indices_held_out,:], targets[indices_held_out]
    )

def calculate_log_loss(logits, targets):
    """Compute log loss."""
    log_softmax_vals = scipy.special.log_softmax(logits, axis=1)
    return LogLoss(log_softmax_vals, targets, priors=None, norm=True)

def calculate_logloss_and_calibrate(full_logits, full_targets, model_name, calibration_prop=0.2):
    ###################################################################################################
    # Create the calibrated scores and the calibration model

    # Choose EPSR (LogLoss or Brier), calibration method, and, optionally, set the priors to the
    # deployment ones, if they are expected to be different from the ones in the test data.

    metric = LogLoss 
    calmethod = AffineCalLogLoss
    deploy_priors = None
    calparams = {'bias': True, 'priors': deploy_priors}
    # Divide data into calibration and held-out sets

    scores_trn, targets_trn, scores_tst, targets_tst = divide_test_and_cal(
        full_logits, full_targets, prop=calibration_prop
    )

    scores_tst_cal, calmodel = calibration_train_on_heldout(scores_tst, scores_trn, targets_trn, 
                                                            calparams=calparams, 
                                                            calmethod=calmethod, 
                                                            return_model=True)

    ###################################################################################################
    # Compute the selected EPSR before and after calibration
    # import pdb
    # pdb.set_trace()
    # calculate_accuracy(scores_tst, targets_tst)
    import scipy
    log_softmax_val_base = scipy.special.log_softmax(scores_tst, axis=1)
    overall_perf = metric(log_softmax_val_base, targets_tst, priors=deploy_priors, norm=True)
    
    overall_perf_after_cal = metric(scores_tst_cal, targets_tst, priors=deploy_priors, norm=True)
    cal_loss = overall_perf-overall_perf_after_cal
    rel_cal_loss = 100*cal_loss/overall_perf

    # print(f"Overall performance before calibration ({metric.__name__}) = {overall_perf:4.2f}" ) 
    # print(f"Overall performance after calibration ({metric.__name__}) = {overall_perf_after_cal:4.2f}" ) 
    # print(f"Calibration loss = {cal_loss:4.2f}" ) 
    # print(f"Relative calibration loss = {rel_cal_loss:4.1f}%" ) 

    ###################################################################################################

    #torch.save(calibration_model.state_dict(), f'models/{model_name}_calibration.pth')
    performance_dict = {
    "overall_perf": overall_perf,
    "overall_perf_after_cal": overall_perf_after_cal,
    "cal_loss": cal_loss,
    "rel_cal_loss": rel_cal_loss
    }

    return performance_dict, scores_tst, targets_tst, scores_tst_cal