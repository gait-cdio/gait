import numpy as np

# Vanilla validation
def validate(ground_truth, estimation):
    diff=np.sum(np.abs(ground_truth-estimation)) #ignores NaN
    weight=np.sum(estimation) # ignores NaN using steps are 1
    return diff/weight

# Land mover distance?