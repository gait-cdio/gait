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