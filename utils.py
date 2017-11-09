import numpy as np

def annotationToOneHot(anno):
    left = 0
    right = 0
    bin = np.zeros((2, anno.size))
    for t in range(anno.size):
        if anno[t] == 1:
            left = 1
        if anno[t] == 2:
            left = 0
        if anno[t] == 3:
            right = 1
        if anno[t] == 4:
            right = 0
        bin[0,t]=left
        bin[1, t] = right
    return bin


def greedy_similarity_match(sim_mat, similarity_threshold):
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

        sim_mat[:, best_match[1]] = np.nan
        sim_mat[best_match[0], :] = np.nan
    return match_list