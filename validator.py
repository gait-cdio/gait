import numpy as np


# Vanilla validation
def validate(ground_truth, estimation):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    weight = np.sum(~np.isnan(diff))
    error_per_ok_frame=float(diff_sum/weight)

    no_annotation = np.isnan(ground_truth)
    no_estimation = np.isnan(estimation)
    false_detections = np.sum(np.logical_and(no_annotation, np.logical_not(no_estimation)))
    missed_detections = np.sum(np.logical_and(np.logical_not(no_annotation), no_estimation))
    false_detection_per_frame = float(false_detections/diff.size)
    missed_detection_per_frame = float(missed_detections/diff.size)
    return {
        'Error per correctly detected frame': error_per_ok_frame,
        'False detections per frame': false_detection_per_frame,
        'Missed detections per frame': missed_detection_per_frame,
    }
