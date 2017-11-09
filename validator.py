import numpy as np


# Vanilla validation
def validate(ground_truth, estimation):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    weight = len(diff) - np.count_nonzero(np.isnan(diff))
    return diff_sum/weight

# Land mover distance?