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

def find_ups(updown):
    """ Take numpy array of up/down values, output indices where the feet leave the ground
    """
    ups = np.transpose(np.where(np.diff(updown) == -1))
    return ups

def find_downs(updown):
    """ Take numpy array of up/down values, output indices where the feet start touching the ground
    """
    downs = np.transpose(np.where(np.diff(updown) == 1))
    return downs

def group_by_foot(indices):
    result = [] 
    for foot in np.unique(indices[:,0]):
        indices_for_foot = indices[np.where(indices[:,0] == foot), 1]
        result.append(indices_for_foot)
    return result

def get_frames(indices):
    return indices[:, 1]

def find_grouped_up_down_indices(updown):
    return {
        'ups': get_frames(find_ups(updown)),
        'downs': get_frames(find_downs(updown)),
    }

def diff_closest(ind_a, ind_b):
    mat1 = np.stack((ind_a,)*len(ind_b), axis=0)
    mat2 = np.stack((ind_b.T,)*len(ind_a), axis=1)

    longest_axis = np.argmax(mat1.shape)

    absmat = np.abs(mat1 - mat2)
    minimum_diffs = np.min(absmat, axis=longest_axis)

    return minimum_diffs

def load_est_and_gt(video_name):
    est = np.load('output-data/{}_updown.npy'.format(video_name))
    gt = np.load('annotations/{}-up_down.npy'.format(video_name))
    return est, gt
