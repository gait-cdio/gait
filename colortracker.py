import cv2
import numpy as np


def match_points(new_points, pointString, similarity_threshold):
    # Simularities between current frame and previous pointstrings
    num_previous_points = len(pointString)
    num_detections = len(new_points)
    if num_detections == 0:
        pass
    # First detection of any points
    elif num_previous_points == 0:
        for point_index in range(0, num_detections):
            newPointList = [new_points[point_index]]
            pointString.append(newPointList)

    # Comparing previous strings and current frame
    else:
        simMat = np.zeros((num_previous_points, num_detections))
        for ix in range(0, num_previous_points):
            for point_index in range(0, num_detections):
                # Only distance at the moment ( 0,0,0 )
                simMat[ix, point_index] = feature_distance(pointString[ix][-1], new_points[point_index], size_weight=0,
                                                           hue_weight=4, time_weight=1)

        while True:
            try:
                best_match_position = np.unravel_index(np.nanargmin(simMat), (num_previous_points, num_detections))
            except ValueError:
                break
            previous_point_index, new_point_index = best_match_position

            if simMat[best_match_position] < similarity_threshold:
                pointString[previous_point_index].append(new_points[new_point_index])
            else:
                newPointString = [new_points[new_point_index]]
                pointString.append(newPointString)

            simMat[:, new_point_index] = np.nan
            simMat[previous_point_index, :] = np.nan


def extract_median_circle(img, xpos, ypos, radius):
    cir=np.zeros(img.shape,np.uint8)
    cv2.circle(cir,center=(xpos,ypos),radius=radius,color=255,thickness=-1)
    return np.median(img[(cir == 255)])


# Low scores if similar
def feature_distance(point1, point2, distance_weight=1, size_weight=0, hue_weight=0, time_weight=0):
    x1, y1 = point1.position
    x2, y2 = point2.position
    d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    s2 = (point1.size - point2.size) ** 2
    h = np.abs(point1.hue - point2.hue)
    t = np.abs(point1.frame - point2.frame)
    s = np.sqrt(s2)
    d = np.sqrt(d2)
    return d * distance_weight + s * size_weight + h * hue_weight + t * time_weight