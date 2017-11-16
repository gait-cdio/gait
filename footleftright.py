import numpy as np
import utils
import itertools

from utils import Direction
from numpy import linalg as LA


def track_fill(tracks):
    num_frames=tracks[0].current_frame + 1
    pos = np.zeros((len(tracks), num_frames, 2))*np.nan
    observed = np.zeros((len(tracks), num_frames), dtype=bool)
    for track_index, track in enumerate(tracks):
        for state in track.state_history:
            pos[track_index,state.frame] = (state.x, state.y)
            observed[track_index,state.frame] = state.observed
    return pos, observed

def track_scorer(track):
    return [state.observed for state in track.state_history].count(True)


def best_tracks(tracks):
    number_of_tracks = len(tracks)
    track_score = np.zeros(number_of_tracks)
    for index, track in enumerate(tracks):
        track_score[index] = track_scorer(track)
    best_track_index = track_score.argsort()[0:number_of_tracks]
    return tracks[best_track_index]


def track_direction(track):
    x = np.array([state.x for state in track.state_history])
    frame = np.array([state.frame for state in track.state_history])
    slope = np.polyfit(frame, x, 1)  # weights here could be useful if start and stop mess it up
    if slope > 1:
        return Direction.right
    return Direction.left


def general_direction(tracks):
    direction_acum = 0
    for index, track in enumerate(tracks):
        # Adds positive/negative values to direction_acum depending on if track_direction is right/left.
        # Amplitude scaled by track_score.
        direction_acum += (2 * (track_direction(track) == Direction.right) - 1) * track_scorer(track)
    if direction_acum > 0:
        return Direction.right
    return Direction.left


def front_foot_scorer(track):
    observed = np.array([state.observed for state in track.state_history])
    observed = np.trim_zeros(observed)
    return np.nansum(observed) / len(observed)


def different_foot_score(pos1, pos2):
    distance = LA.norm((pos1 - pos2), axis=1)
    mean = np.nanmean(distance)
    variance = np.nanvar(distance)
    return variance / mean


def group_tracks_by_feet(tracks):
    n_tracks = len(tracks)
    association_matrix=np.zeros((n_tracks,n_tracks))*np.nan
    for index1, track1 in enumerate(tracks):
        for index2 in range(index1 + 1, n_tracks):
            track2=tracks[index2]
            association_matrix[index1,index2] = different_foot_score(track1, track2)

    matches = utils.greedy_similarity_match(association_matrix, similarity_threshold=1)

    unmatched_indices = list(set(range(n_tracks)) - set(itertools.chain(*matches)))

    return matches + [(singleton_index,) for singleton_index in unmatched_indices]


if __name__ == '__main__':
    tracks = np.load("TrackerResults/4fargersilly.mp4.tracks.npy")

    tracks, observed = track_fill(tracks)
    matches = group_tracks_by_feet(tracks)
    print(matches)