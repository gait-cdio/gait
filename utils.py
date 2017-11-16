import numpy as np
from enum import Enum

def annotationToUpDown(annotations):
    """ Convert ground truth annotations to a matrix of 1s and 0s.
    The first row of the matrix is the left foot,
    the second is the right foot.
    A value of 1 means the foot is up, 0 means it is down.

    :param annotations:
    :return: np.ndarray
    """
    left = 0
    right = 0
    bin = np.zeros((2, annotations.size))
    for t in range(annotations.size):
        if annotations[t] == 1:
            left = 1
        if annotations[t] == 2:
            left = 0
        if annotations[t] == 3:
            right = 1
        if annotations[t] == 4:
            right = 0
        bin[0, t] = left
        bin[1, t] = right
    return bin

class Direction(Enum):  # general_direction is using the representation of numbers at the moment
    right = 1
    left = 0

def greedy_similarity_match(input_mat, similarity_threshold):
    sim_mat = np.copy(input_mat)
    # Get match list greedy by always picking the minimum distance
    match_list = []
    rows, cols = sim_mat.shape
    while True:
        try:
            best_match = np.unravel_index(np.nanargmin(sim_mat), (rows, cols))
        except ValueError:
            break

        similarity_score = sim_mat[best_match]
        if similarity_score < similarity_threshold:
            match_list.append(best_match)

        row, col = best_match
        sim_mat[:, col] = np.nan
        sim_mat[row, :] = np.nan
    return match_list


def load_groundtruth(filename):
    footstates = np.load(filename)
    return annotationToUpDown(footstates)
