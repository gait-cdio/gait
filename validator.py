import numpy as np


# Vanilla validation
def validate(ground_truth, estimation):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    weight = diff.shape[1] - np.count_nonzero(np.isnan(diff))
    error_per_ok_frame=float(diff_sum/weight)

    no_annotation = np.isnan(ground_truth)
    no_estimation = np.isnan(estimation)
    false_alarms = np.sum(np.logical_and(no_annotation, np.logical_not(no_estimation)))
    missed_detections = np.sum(np.logical_and(np.logical_not(no_annotation), no_estimation))
    false_alarm_per_frame = float(false_alarms/diff.shape[1])
    missed_detection_per_frame = float(missed_detections/diff.shape[1])
    return {'Error per correctly detected frame': error_per_ok_frame, 'False alarm per frame': false_alarm_per_frame,
            'Missed detection per frame': missed_detection_per_frame}


def error(ground_truth, estimation, nan_penalty):
    diff = np.abs(ground_truth - estimation)
    diff_sum = np.nansum(diff)
    numNaN=np.count_nonzero(np.isnan(estimation))
    return (diff_sum + nan_penalty * numNaN)/len(ground_truth)

# Land mover distance?
