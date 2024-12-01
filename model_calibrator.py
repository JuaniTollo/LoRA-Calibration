import numpy as np
import torch
from expected_cost.calibration import calibration_train_on_heldout
from expected_cost.psrcal_wrappers import LogLoss
from psrcal.calibration import AffineCalLogLoss

def calculate_log_loss(test_logits, test_targets):
    import scipy
    log_softmax_vals = scipy.special.log_softmax(test_logits, axis=1)
    overall_perf = LogLoss(log_softmax_vals, test_targets, priors=None, norm=True)
    print(f"Overall performance ({LogLoss.__name__}) = {overall_perf:.2f}")
    return overall_perf

def calibrate_and_evaluate(scores_tst, scores_trn, targets_trn, targets_tst, model_name):
    overall_perf = calculate_log_loss(scores_tst, targets_tst)

    # Calibrate base logits
    scores_tst_cal, cal_model = calibration_train_on_heldout(
        scores_tst, scores_trn, targets_trn,
        calparams={'bias': True, 'priors': None},
        calmethod=AffineCalLogLoss,
        return_model=True
    )
    torch.save(cal_model.state_dict(), f'models/{model_name}.pth')
    overall_perf_after_cal = LogLoss(scores_tst_cal, targets_tst, priors=None, norm=True)

    # Compute calibration loss metrics
    cal_loss = overall_perf - overall_perf_after_cal
    rel_cal_loss = 100 * cal_loss / overall_perf

    return {
        "overall_perf": overall_perf,
        "overall_perf_after_cal": overall_perf_after_cal,
        "cal_loss": cal_loss,
        "rel_cal_loss": rel_cal_loss,
        "calibrated_logits_base": scores_tst_cal,  # Add calibrated logits
        "calibrated_logits_fine_tuned": scores_tst_cal,  # Add fine-tuned logits if applicable
    }

