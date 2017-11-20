import numpy as np
import utils
import itertools

from utils import Direction


def track_fill(tracks):
    num_frames = tracks[0].current_frame + 1
    num_tracks = len(tracks)
    pos = np.zeros((num_tracks, num_frames, 2)) * np.nan
    observed = np.zeros((len(tracks), num_frames), dtype=bool)
    for track_index, track in enumerate(tracks):
        for state in track.state_history:
            pos[track_index, state.frame] = (state.x, state.y)
            observed[track_index, state.frame] = state.observed
    return pos, observed


def track_direction(track):
    # TODO(rolf): remove NaNs from x, either here or in some calling function
    x = track[:, 0]
    coefficients = np.polyfit(range(len(x)), x, deg=1)  # weights here could be useful if start and stop mess it up
    if coefficients[0] > 0:
        return Direction.right
    elif coefficients[0] < 0:
        return Direction.left
    else:
        raise ValueError('Undefined slope! Does the track contain NaNs?', x)


def general_direction(tracks):
    direction_acum = 0
    num_tracks = tracks.shape[0]
    for track_index in range(num_tracks):
        # Adds positive/negative values to direction_acum depending on if track_direction is right/left.
        direction_acum += (2 * (track_direction(tracks[track_index]) == Direction.right) - 1)
    if direction_acum > 0:
        return Direction.right
    elif direction_acum < 0:
        return Direction.left
    else:
        raise ValueError("Inconclusive results from direction estimation. We can't have that!")


def front_foot_scorer(observed):
    n_tracks = observed.shape[0]
    scores = np.zeros(n_tracks)
    for tracks_index in range(n_tracks):
        trimmed = np.trim_zeros(observed[tracks_index, :])
        scores[tracks_index] = np.nansum(trimmed) / len(trimmed)
    return scores


def length_scorer(observed):
    n_tracks = observed.shape[0]
    scores = np.zeros(n_tracks)
    for tracks_index in range(n_tracks):
        trimmed = np.trim_zeros(observed[tracks_index, :])
        scores[tracks_index] = np.nansum(trimmed) / observed.shape[1]
    return scores


def different_foot_score(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), axis=1)
    mean = np.nanmean(distance)
    variance = np.nanvar(distance)
    return variance / mean


def group_tracks_by_feet(tracks):
    n_tracks = len(tracks)
    association_matrix = np.zeros((n_tracks, n_tracks)) * np.nan
    for index1, track1 in enumerate(tracks):
        for index2 in range(index1 + 1, n_tracks):
            track2 = tracks[index2]
            association_matrix[index1, index2] = different_foot_score(track1, track2)

    matches = utils.greedy_similarity_match(association_matrix, similarity_threshold=5)

    unmatched_indices = list(set(range(n_tracks)) - set(itertools.chain(*matches)))

    return matches + [(singleton_index,) for singleton_index in unmatched_indices]


def front_foot_back_foot(matches, front_foot_score):
    # Returns a list with two tuples. First tuple is front foot, last is back foot.
    scores = np.zeros(2)
    for foot_index, foot in enumerate(matches):
        for point in foot:
            scores[foot_index] += front_foot_score[point] / len(foot)
    return [matches[np.argmax(scores)], matches[np.argmin(scores)]]


def travel_scorer(tracks):
    # Check only the x-distance traveled. 3D tracks indexed with 0 in z gives x values
    dist_traveled = np.nanmax(tracks[:, :, 0], axis=1) - np.nanmin(tracks[:, :, 0], axis=1)
    norm_dist_traveled = dist_traveled / np.max(dist_traveled)
    return norm_dist_traveled


def rate_that_foot(matches, tracks, observed, len_weight):
    num_pairs = len(matches)
    num_frames = tracks.shape[1]
    scores = np.zeros((num_pairs))

    travel_scores = travel_scorer(tracks)
    length_scores = length_scorer(observed)
    total_scores = travel_scores + length_scores * len_weight
    for pair_index, pair in enumerate(matches):
        for point in pair:
            scores[pair_index] += total_scores[point] / len(pair)
    best_feet_index = ((-scores).argsort()[0:2])
    return [matches[pair] for pair in best_feet_index]


def toe_point_heel_point(matches, tracks, dir):
    for pair_index, pair in enumerate(matches):
        if len(pair) > 1:
            # less than 0 if first point in pair is more to the left
            first_in_pair_left = 0 > np.sum(tracks[pair[0], :, 0] - tracks[pair[1], :, 0])
            # != is used as a xor
            if (dir == Direction.left) != first_in_pair_left:
                matches[pair_index] = (pair[1], pair[0])
    return matches


# This is the main function: Returns list of tuples describing track indexes. First element leftfoot, second right foot
# If multiple elements in tuple, the first one is the toe.
def left_foot_right_foot(track_input):
    tracks, observed = track_fill(track_input)
    matches = group_tracks_by_feet(tracks)
    front_scores = front_foot_scorer(observed)
    best_scores = rate_that_foot(matches, tracks, observed, 1)
    dir = general_direction(tracks)
    ordered_feet = front_foot_back_foot(best_scores, front_scores)
    ordered_feet_with_toe_heel = toe_point_heel_point(ordered_feet, tracks, dir)
    if dir == Direction.left:
        return ordered_feet_with_toe_heel, dir
    elif dir == Direction.right:
        return reversed(ordered_feet_with_toe_heel), dir
    else:
        ValueError('Unknown direction', dir)


if __name__ == '__main__':
    tracks = np.load("TrackerResults/4farger.mp4.tracks.npy")
    print(left_foot_right_foot(tracks))
