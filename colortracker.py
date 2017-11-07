import cv2
import numpy as np
from recordclass import recordclass


def variance_greater_than(threshold):
    def fun(track):
        x = list(map(lambda point: point.position[0], track))
        x_variance = np.var(x)

        return x_variance > threshold

    return fun


class ColorTracker:
    def __init__(self, median_filter=True):
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.minThreshold = 10
        self.blob_params.maxThreshold = 180
        self.blob_params.thresholdStep = 20
        self.blob_params.minRepeatability = 1
        self.blob_params.filterByCircularity = False
        self.blob_params.minCircularity = 0.5
        self.blob_params.filterByInertia = True
        self.blob_params.minInertiaRatio = 0.3
        self.blob_params.minDistBetweenBlobs = 10
        self.blob_params.filterByArea = True
        self.blob_params.minArea = 60
        self.blob_params.maxArea = 50000
        self.blob_params.filterByConvexity = True
        self.blob_params.minConvexity = 0.9
        self.blob_params.filterByColor = 0
        self.blob_params.blobColor = 100

        self.detector = cv2.SimpleBlobDetector_create(self.blob_params)

        self.median_filter = median_filter

        # This seems to capture blue and pink
        self.hsv_min = (80, 0, 0)
        self.hsv_max = (180, 255, 255)

        self.gaussian_kernel_size = (15, 15)
        self.gaussian_kernel_sigma = 1

        self.visualize_keypoints = True
        self.visualize_blurred_masked = False

        self.variance_threshold = 20 ** 2

    def detect(self, img, frame_nr):
        if self.median_filter:
            blurred_img = cv2.medianBlur(img, 3)
            hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_min, self.hsv_max)

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        blurred_masked = cv2.GaussianBlur(masked_img,
                                          self.gaussian_kernel_size,
                                          self.gaussian_kernel_sigma)

        keypoints = self.detector.detect(blurred_masked)

        if self.visualize_keypoints:
            visualize_detections(img, keypoints)
        if self.visualize_blurred_masked:
            visualize_detections(blurred_masked, keypoints, window_title='Blurred masked')

        return list(map(lambda keypoint: PointFeatures(
            position=keypoint.pt,
            size=keypoint.size,
            hue=float(hsv[int(keypoint.pt[1]), int(keypoint.pt[0]), 0]),
            frame=frame_nr
        ), keypoints))

    def cleanup_windows(self):
        cv2.destroyWindow('Blurred masked')
        cv2.destroyWindow('Keypoints')


def visualize_detections(img, keypoints, window_title='Keypoints'):
    im_with_keypoints = cv2.drawKeypoints(img,
                                          keypoints,
                                          np.array([]),
                                          (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title,800,600)
    cv2.imshow(window_title, im_with_keypoints)


PointFeatures = recordclass('PointFeatures', ['position', 'size', 'hue', 'frame'])


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
