
import numpy as np
import matplotlib.pyplot as plt

def visualize_gait(gait, fig=None):
    n_gaits = len(gait)
    n_frames = len(gait[0])
    gait_diff = np.diff(gait)

    gait_cycle = []
    for track_index, track in enumerate(gait_diff):
        start = None
        end = None
        gait_cycle.append([])
        # Extract up and down indices
        for frame, chg in enumerate(track):
            if chg == 1:
                start = frame
            if chg == -1:
                if not start: continue
                end = frame
                gait_cycle[track_index].append((start, end-start))

    if not fig:
        fig, ax = plt.subplots()
    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, 2*n_gaits+1)
    ax.set_xlabel('Frame')

    for cycle_index, cycle in enumerate(gait_cycle):
        ax.broken_barh(cycle, (2*cycle_index+1, 1), facecolors='black')

    return fig

    
    

