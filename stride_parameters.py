import yaml
import numpy as np
import math


def write_gait_parameters_to_file(filename, updown_estimations, fps):
    parameters = {}
    parameters['duty_cycle'] = {}
    parameters['stance_duration'] = {}
    parameters['swing_duration'] = {}
    feet_order=['left', 'right']

    for i, updown in enumerate(updown_estimations):
        duty_c, stance_d, swing_d = duty_stance_swing(updown, fps)
        parameters['duty_cycle'][feet_order[i]] = duty_c
        parameters['stance_duration'][feet_order[i]] = stance_d
        parameters['swing_duration'][feet_order[i]] = swing_d

    with open(filename, 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)


def duty_stance_swing(updown_with_nans, fps):
    # Remove all values when foot is not in frame (i.e. nan values)
    updown = updown_with_nans[~np.isnan(updown_with_nans)]

    # Remove first incomplete steps, only consider values between first and last
    # transition. 
    first_transition = np.where(np.diff(updown) != 0)[0][0] + 1
    last_transition = np.where(np.diff(updown) != 0)[0][-1] + 1
    clean_updown = updown[first_transition:last_transition]

    # Calculate how many up cycles and down cycles there are
    n_up = sum(np.diff(clean_updown) == -1) + (clean_updown[0] == 0)
    n_down = sum(np.diff(clean_updown) == 1) + (clean_updown[0] == 1)

    # Count how many frames the foot is in the air/on the ground
    n_up_frames = sum(clean_updown == 0)
    n_down_frames = sum(clean_updown == 1)

    try:
        # Count the mean number of frames the foot is up/down per step
        mean_frames_per_up = n_up_frames / n_up
        mean_frames_per_down = n_down_frames / n_down

        # Calculate the duty cycle = time_foot_is_down / total_time
        duty_cycle = mean_frames_per_down / (mean_frames_per_down + mean_frames_per_up)
        duty_cycle = float(duty_cycle)
        if math.isnan(duty_cycle):
            duty_cycle = 'Too few steps to confidently calculate duty cycle'
    except ZeroDivisionError:
        duty_cycle = 'Too few steps to confidently calculate duty cycle'

    try:
        # Calculate mean stance duration
        stance_duration = float(n_down_frames) / float((n_down) * fps)
        stance_duration = float(stance_duration)
    except ZeroDivisionError:
        stance_duration = 'Too few steps to confidently calculate stance duration'
    try:
        # Calculate mean swing duration
        swing_duration = float(n_up_frames) / float((n_up) * fps)
        swing_duration = float(swing_duration)
    except ZeroDivisionError:
        swing_duration = 'Too few steps to confidently calculate swing duration'

    return duty_cycle, stance_duration, swing_duration
