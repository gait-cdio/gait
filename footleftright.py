import numpy as np
from enum import Enum


class Direction(Enum):  # general_direction is using the representation of numbers at the moment
    right = 1
    left = 0


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


def general_direction(tracks) -> Direction:
    sum = 0
    for index, track in enumerate(tracks):
        sum += (2 * (track_direction(track) == Direction.right) - 1) * track_scorer(track)
    if sum > 0:
        return Direction.right
    return Direction.left


def front_foot_scorer(track):
    observed = np.array([state.observed for state in track.state_history])
    observed = np.trim_zeros(observed)
    return np.sum(observed) / len(observed)


def four_point_sorter(tracks):
    direction = general_direction(tracks)

    number_of_tracks = len(tracks)
    front_score = np.zeros(number_of_tracks)
    for index, track in enumerate(tracks):
        front_score[index] = front_foot_scorer(track)

    sorted_indices = front_score.argsort()
    front_tracks = sorted_indices[0:2]
    back_tracks = sorted_indices[2:4]
    ...
