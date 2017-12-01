import yaml
import numpy as np


def write_gait_parameters_to_file(filename, updown_estimations, fps):
    parameters = {}
    parameters['duty_cycle'] = {}
    feet_order=['left', 'right']

    for i, updown in enumerate(updown_estimations):
        parameters['duty_cycle'][feet_order[i]] = duty_cycle(updown)
        stance_duration(updown, fps)
        swing_duration(updown, fps)

    with open(filename, 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)


def duty_cycle(updown_with_nans):
    # Remove all values when foot is not in frame (i.e. nan values)
    updown = updown_with_nans[~np.isnan(updown_with_nans)]

    # Remove first incomplete steps, only consider values between first and last
    # transition. 
    first_transition = np.where(np.diff(updown) != 0)[0][0] + 1
    last_transition = np.where(np.diff(updown) != 0)[0][-1] + 1
    clean_updown = updown[first_transition:last_transition]

    # Calculate how many times the foot goes up respictevely down
    n_up = sum(np.diff(clean_updown) == -1)
    n_down = sum(np.diff(clean_updown) == 1)

    # Count how many frames the foot is in the air/on the ground
    n_up_frames = sum(clean_updown == 0)
    n_down_frames = sum(clean_updown == 1)

    # Count the mean number of frames the foot is up/down per step
    mean_frames_per_up = n_up_frames / n_up
    mean_frames_per_down = n_down_frames / n_down

    # Calculate the duty cycle = time_foot_is_down / total_time
    duty_cycle = mean_frames_per_down / (mean_frames_per_down + mean_frames_per_up)

    return float(duty_cycle)

def stance_duration(updown_with_nans, fps):
    # Remove all values when foot is not in frame (i.e. nan values)
    updown = updown_with_nans[~np.isnan(updown_with_nans)]

    # Remove first incomplete steps, only consider values between first and last
    # transition.
    first_transition = np.where(np.diff(updown) != 0)[0][0] + 1
    last_transition = np.where(np.diff(updown) != 0)[0][-1] + 1
    clean_updown = updown[first_transition:last_transition]

    # Calculate how many times the foot goes up respictevely down
    n_up = sum(np.diff(clean_updown) == -1)
    n_down = sum(np.diff(clean_updown) == 1)

    # Count how many frames the foot is in the air/on the ground
    n_up_frames = sum(clean_updown == 0)
    n_down_frames = sum(clean_updown == 1) # will show one less than the amount of 'down' cycles, as the foot starts down in clean_updown

    # Calculate mean stance duration
    stance_duration = float(n_down_frames) / float((n_down + 1) * fps)

    return float(stance_duration)

def swing_duration(updown_with_nans, fps):
    # Remove all values when foot is not in frame (i.e. nan values)
    updown = updown_with_nans[~np.isnan(updown_with_nans)]

    # Remove first incomplete steps, only consider values between first and last
    # transition.
    first_transition = np.where(np.diff(updown) != 0)[0][0] + 1
    last_transition = np.where(np.diff(updown) != 0)[0][-1] + 1
    clean_updown = updown[first_transition:last_transition]

    # Calculate how many times the foot goes up respictevely down
    n_up = sum(np.diff(clean_updown) == -1)
    n_down = sum(np.diff(clean_updown) == 1)

    # Count how many frames the foot is in the air/on the ground
    n_up_frames = sum(clean_updown == 0)
    n_down_frames = sum(clean_updown == 1)

    # Calculate mean swing duration
    swing_duration = float(n_up_frames) / float((n_up) * fps)

    return float(swing_duration)