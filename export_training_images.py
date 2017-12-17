import numpy as np
import cv2

# coding: utf-8
def write_all_the_images():
    return [
        [
            np.all([
                cv2.imwrite("output-images/training-data/{}-{}-{:04}.png".format(name, direction, index),
                            ((direction == 'output') * 254 + 1) * image)
                for index, image in enumerate(np.load("annotations/{}_markerless-positions.npy_{}_cache.npy".format(name, direction)))
            ])
            for name in ['kevin', 'john', 'rolf']
        ]
        for direction in ['input', 'output']
    ]
