import yaml
import numpy as np

def write_gait_parameters_to_file(filename, updown_estimations):
    parameters = {}
    parameters['duty_cycle'] = {}

    for i, updown in enumerate(updown_estimations):
        parameters['duty_cycle'][i] = str(duty_cycle(updown)) + " <-- this value is incorrect!"
    
    with open(filename, 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)


def duty_cycle(updown_with_nans):
    """ NOTE THIS FUNCTION DOES NOT CALCULATE THE DUTY CYCLE CORRECTLY """
    # Remove all values when foot is not in frame (i.e. nan values)
    updown  = updown_with_nans[~np.isnan(updown_with_nans)]

    # Remove first zeros before walk has started
    updown  = np.trim_zeros(updown)
    toe_up    = np.where(np.diff(updown) == 1)[0]
    heel_down = np.where(np.diff(updown) == -1)[0]

    n_all   = updown.size
    n_down  = n_all - np.count_nonzero(updown)
    
    duty_cycle = n_down / n_all
    return duty_cycle

