import cv2
import numpy as np
from recordclass import recordclass
import pickle
import os

from threshold_selector import set_threshold

TrackerWithDetections = recordclass('TrackerWithDetections', ['tracker', 'detections'])

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

    def detect(self, img, frame_nr, visualize = True):

        self.visualize_keypoints = visualize

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
        cv2.destroyAllWindows()


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
def feature_distance(distance_weight=1, size_weight=0, hue_weight=0, time_weight=0):
    def closure(point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        s2 = (point1.size - point2.size) ** 2
        h = np.abs(point1.hue - point2.hue)
        t = np.abs(point1.frame - point2.frame)
        s = np.sqrt(s2)
        d = np.sqrt(d2)
        return d * distance_weight + s * size_weight + h * hue_weight + t * time_weight
    return closure

def detect(filename, number_of_trackers=1, visualize=False, set_thresholds=True):
    cap = cv2.VideoCapture('input-videos/' + filename)
    number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # +-----------------------------------------------------------------------+
    # |                         Initialize trackers                           |
    # +-----------------------------------------------------------------------+
    # Initialize stuff
    # Select marker/-less
    # Maybe prompt user to select colors, regions, etc.
    # Create instances of necessary classes (SimpleBlobDetector, TrackerKCF, etc.)
    trackerList = []

    for i in range(number_of_trackers):
        # Check if there are any saved values for thresholds and load it
        basename = os.path.splitext(os.path.basename(filename))[0]
        threshold_file = ('hsv-threshold-settings/' +
                          basename + '-' + 'threshold' + str(i) + '.pkl')
        try:
            with open(threshold_file, 'rb') as f:
                default_thresholds = pickle.load(f)
        except FileNotFoundError:
            if not set_thresholds:
                print("Error: No saved thresholds found for " + filename)
                print("       Please set the thresholds for " + filename)
                print("       before running with choose_new_thresholds = False")
            else:
                default_thresholds = None

        if(set_thresholds):
            thresholds = set_threshold(cap, default_thresholds)

            # Save threshold settings
            with open(threshold_file, 'wb') as f:
                pickle.dump(thresholds, f)

        keypoint_tracker = ColorTracker()
        keypoint_tracker.hsv_min, keypoint_tracker.hsv_max = thresholds
        trackerList.append(TrackerWithDetections(tracker=keypoint_tracker, detections=[]))

    paused = False

    # +-----------------------------------------------------------------------+
    # |                          Detect keypoints                             |
    # +-----------------------------------------------------------------------+
    # Detect keypoints (heel, toe) in each frame
    for frame_nr in range(number_frames):
        cap.set(1, frame_nr)
        ret, img = cap.read()

        # cv2.putText(img, str(frame_nr), (10, 30), fontFace=font, fontScale=1, color=(0, 0, 255))

        for tracker in trackerList:
            detections_for_frame = tracker.tracker.detect(img, frame_nr, visualize=visualize)
            tracker.detections.append(detections_for_frame)

        if visualize:
            if paused:
                delay = 0
            else:
                delay = 1

            pressed_key = cv2.waitKey(delay) & 0xFF
            if pressed_key == ord(' '):
                paused = not paused
            elif pressed_key == ord('q'):
                break

    # +-----------------------------------------------------------------------+
    # |                         Handle missing frames                         |
    # +-----------------------------------------------------------------------+
    for tracker in trackerList:
        tracker.tracker.cleanup_windows()
        missing_frames = max(number_frames - len(tracker.detections), 0)
        tracker.detections += [[]] * missing_frames

    return list(map(lambda tracker: tracker.detections, trackerList))
