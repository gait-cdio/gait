import numpy as np
import scipy.ndimage.filters as filt
from scipy import signal


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


def estimate_naive(tracks, max_frame, applicability_std=1.3, blur_std=3, up_thresh=0.25, down_thresh=0.45):
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
        dx2_c = filt.gaussian_filter1d(input=fixed_x, sigma=blur_std, order=2, mode='nearest')  # second derivative
        #thresh = np.where(dx_c < (np.min(dx_c) * speed_thresh))
        est_step = np.zeros(dx_c.shape)
        #est_step[thresh] = 1
        dx_peak = np.min(dx_c)

        # The block below does a sequential sweep to determine whether the foot is up/down. Supports different thresholds for up/down
        est_step[0] = int(dx_c[0] < (dx_peak * up_thresh) and dx2_c[0] < 0)
        for i in range(1, len(dx_c)):
            if est_step[i - 1] == 0:
                est_step[i] = int(dx_c[i] < (dx_peak * up_thresh) and dx2_c[i] < 0)
            else:
                est_step[i] = int(dx_c[i] < (dx_peak * down_thresh) or dx2_c[i] < 0)

        start_padding = np.ones(frame_offset) * np.nan
        end_padding = np.ones(max_frame-len(est_step) - frame_offset) * np.nan
        est_step = np.concatenate((start_padding, est_step, end_padding))
        dx_c = np.concatenate((start_padding, dx_c, end_padding))

        estimations.append(est_step)
        derivatives.append(dx_c)

    return estimations, derivatives

def estimate_detrend(tracks, max_frame, applicability_std=1.3, blur_std=0.5, up_thresh=0.25, down_thresh=0.45):
    estimations = []
    derivatives = []

    for index, track in enumerate(tracks):
        t_c = [state.frame for state in track.state_history]
        x_c = signal.detrend([state.x for state in track.state_history])
        y_c = signal.detrend([state.y for state in track.state_history])
        fixed_x, frame_offset = inpaint_1d(np.array(x_c), np.array(t_c), appStd=applicability_std)
        fixed_y, _ = inpaint_1d(np.array(y_c), np.array(t_c), appStd=applicability_std)
        dx_c = filt.gaussian_filter1d(input=fixed_x, sigma=blur_std, order=1,
                                      mode='nearest')  # order=1 lagpass + derivering. TODO explore mode options
        thresh = np.where(dx_c < 0)
        est_step = np.zeros(dx_c.shape)
        est_step[thresh] = 1

        start_padding = np.ones(frame_offset) * np.nan
        end_padding = np.ones(max_frame - len(est_step) - frame_offset) * np.nan
        est_step = np.concatenate((start_padding, est_step, end_padding))
        dx_c = np.concatenate((start_padding, dx_c, end_padding))

        estimations.append(est_step)
        derivatives.append(dx_c)

    return estimations, derivatives