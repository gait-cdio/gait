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
        'Mean error per up/down transition': float(mean_frames_error(estimation=estimation, groundtruth=ground_truth)),
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


def find_up_down_indices(updown):
    return {
        'ups': get_frames(find_ups(updown)),
        'downs': get_frames(find_downs(updown)),
    }


def diff_closest(ind_a, ind_b):
    """ Calculate the absolute difference of best fit between two 1d arrays

    Smallest difference between each in one array to any in the other
    """
    mat1 = np.stack((ind_a,)*len(ind_b), axis=0)
    mat2 = np.stack((ind_b.T,)*len(ind_a), axis=1)

    shortest_axis = np.argmin(mat1.shape)

    absmat = np.abs(mat1 - mat2)
    minimum_diffs = np.min(absmat, axis=shortest_axis)

    return minimum_diffs


def load_est_and_gt(video_name):
    est = np.load('output-data/{}_updown.npy'.format(video_name))
    gt = np.load('annotations/{}-up_down.npy'.format(video_name))
    return est, gt


def mean_frames_error(estimation, groundtruth):
    indices_est = find_up_down_indices(estimation)
    indices_gt = find_up_down_indices(groundtruth)

    up_diffs = diff_closest(indices_est['ups'], indices_gt['ups'])
    down_diffs = diff_closest(indices_est['downs'], indices_gt['downs'])

    mean_error = (sum(up_diffs) + sum(down_diffs)) / (len(up_diffs) + len(down_diffs))
    return mean_error


if __name__ == '__main__':
    mean_frames_error(*load_est_and_gt('4markersjohn1'))
