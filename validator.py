import numpy as np


# Vanilla validation
def validate(ground_truth, estimation):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    weight = diff.shape[1] - np.count_nonzero(np.isnan(diff))
    return float(diff_sum/weight)


def error(ground_truth, estimation, nan_penalty):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    numNaN=np.count_nonzero(np.isnan(estimation))
    return (diff_sum + nan_penalty * numNaN)/len(ground_truth)

# Land mover distance?
