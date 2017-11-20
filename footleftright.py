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
    x = track[:, 0]
    x = x[~np.isnan(x)]
    coefficients = np.polyfit(range(len(x)), x, deg=1)  # weights here could be useful if start and stop mess it up
    if coefficients[0] > 0:
        return Direction.right
    elif coefficients[0] < 0:
        return Direction.left
    else:
        raise ValueError('Zero or undefined slope!', x)


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


def different_foot_score(pos1, pos2, both_observed):
    distance = np.linalg.norm((pos1 - pos2), axis=1)
    distance[~both_observed] = np.nan
    mean = np.nanmean(distance)
    variance = np.nanvar(distance)
    return variance / mean


def group_tracks_by_feet(tracks, observed):
    n_tracks = len(tracks)
    association_matrix = np.zeros((n_tracks, n_tracks)) * np.nan
    for index1, track1 in enumerate(tracks):
        for index2 in range(index1 + 1, n_tracks):
            track2 = tracks[index2]
            both_observed = np.logical_and(observed[index1], observed[index2])
            association_matrix[index1, index2] = different_foot_score(track1, track2, both_observed)

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
    feet = [None] * len(matches)
    for pair_index, pair in enumerate(matches):
        if len(pair) > 1:
            # less than 0 if first point in pair is more to the left
            first_in_pair_left = np.nanmean(tracks[pair[0], :, 0] - tracks[pair[1], :, 0]) < 0
            # != is used as a xor
            if (dir == Direction.left) != first_in_pair_left:
                toe_index, heel_index = 1, 0
            else:
                toe_index, heel_index = 0, 1

            feet[pair_index] = {
                'toe': pair[toe_index],
                'heel': pair[heel_index]
            }
        else:
            feet[pair_index] = {
                'foot': pair[0]
            }
    return feet


def left_foot_right_foot(track_input):
    """ Take a bunch of tracks as input and return a dict describing which index corresponds to which point on which foot.
    Also estimate which direction the person is walking in.

    :rtype: dict
    :param track_input:
    :return: description of which index corresponds to which point on which foot
    """
    tracks, observed = track_fill(track_input)
    matches = group_tracks_by_feet(tracks, observed)
    front_scores = front_foot_scorer(observed)
    best_scores = rate_that_foot(matches, tracks, observed, 1)
    dir = general_direction(tracks)
    ordered_feet = front_foot_back_foot(best_scores, front_scores)
    ordered_feet_with_toe_heel = toe_point_heel_point(ordered_feet, tracks, dir)
    if dir == Direction.left:
        left_foot = ordered_feet_with_toe_heel[0]
        right_foot = ordered_feet_with_toe_heel[1]
    elif dir == Direction.right:
        left_foot = ordered_feet_with_toe_heel[1]
        right_foot = ordered_feet_with_toe_heel[0]
    else:
        raise ValueError('Unknown direction', dir)

    return {
        'movement_direction': dir,
        'left_foot': left_foot,
        'right_foot': right_foot
    }


if __name__ == '__main__':
    tracks = np.load("TrackerResults/4farger.mp4.tracks.npy")
    print(left_foot_right_foot(tracks))
