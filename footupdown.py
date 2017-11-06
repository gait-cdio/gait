import numpy as np
import scipy.ndimage.filters as filt


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def inpaint_1d(curve, indexes, appStd):
    newLength = np.max(indexes) - np.min(indexes) + 1
    filled = np.zeros(newLength)
    certainty = np.zeros(newLength)
    indOffset = np.min(indexes)

    filled[indexes - indOffset] = curve
    newFilled = np.copy(filled)
    certainty[indexes - indOffset] = 1

    uncertain_points = np.where(certainty == 0)[0]
    if len(uncertain_points) > 0:
        for ii in np.nditer(uncertain_points):
            applicability = gaussian(np.arange(0, newLength), mu=ii, sig=appStd)
            newFilled[ii] = np.sum(applicability * certainty * filled) / np.sum(certainty * applicability)

    return newFilled, indOffset


def estimate_naive(tracks, max_frame, applicability_std=1.3, blur_std=2, speed_thresh=0.5):
    """ Estimate foot up/down vector for each track in tracks.

    :param tracks:
    :param applicability_std: Stddev for applicability function for infill
    :param blur_std: Stddev for derivative
    :param speed_thresh: Multiplier of minimum speed considered to be off the ground compared to max speed
    """
    estimations=[]
    derivatives=[]

    for index, track in enumerate(tracks):
        t_c = [state.frame for state in track.state_history]
        x_c = [state.x for state in track.state_history]
        y_c = [state.y for state in track.state_history]
        fixed_x, frame_offset = inpaint_1d(np.array(x_c), np.array(t_c), appStd=applicability_std)
        fixed_y, _ = inpaint_1d(np.array(y_c), np.array(t_c), appStd=applicability_std)
        dx_c = filt.gaussian_filter1d(input=fixed_x, sigma=blur_std, order=1, mode='nearest')  # order=1 lagpass + derivering. TODO explore mode options

        thresh = np.where(dx_c < (np.min(dx_c) * speed_thresh))
        est_step = np.zeros(dx_c.shape)
        est_step[thresh] = 1

        start_padding = np.ones(frame_offset) * np.nan
        end_padding = np.ones(max_frame-len(est_step) - frame_offset) * np.nan
        est_step = np.concatenate((start_padding, est_step, end_padding))
        dx_c = np.concatenate((start_padding, dx_c, end_padding))

        estimations.append(est_step)
        derivatives.append(dx_c)

    return estimations, derivatives
