import cv2
import numpy as np

from colortracker import ColorTracker
import matplotlib.pyplot as plt

plt.ioff()

# Load videostream
# video_stream = load_video() | stream_from_webcam()

filename = '4farger.mp4'  # TODO: parse arguments for this
cap = cv2.VideoCapture(filename)

# Initialize stuff
# Select marker/-less
# Maybe prompt user to select colors, regions, etc.
# Create instances of necessary classes (SimpleBlobDetector, TrackerKCF, etc.)
keypoint_tracker = ColorTracker()

paused = False

# Detect keypoints (heel, toe) in each frame
detections = []
frame_nr = 0
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        detections_for_frame = keypoint_tracker.detect(img, frame_nr)
        detections.append(detections_for_frame)

        if paused:
            delay = 0
        else:
            delay = 1

        pressed_key = cv2.waitKey(delay) & 0xFF
        if pressed_key == ord(' '):
            paused = not paused
        elif pressed_key == ord('q'):
            break
        frame_nr += 1
    else:
        keypoint_tracker.cleanup_windows()
        break

# Associate keypoints to form tracks
tracks = keypoint_tracker.associate(detections)

# TODO(rolf): make this plotting code prettier

f, axes = plt.subplots(ncols=2)

for index in range(0, len(tracks)):
    curve = tracks[index]
    t_c=[p.frame for p in curve]
    x_c=[p.position[0] for p in curve]
    y_c=[p.position[1] for p in curve]
    axes[0].plot(t_c, x_c, 'o-', markersize=2)
    axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()

# TODO(rolf): link the subplots in some way to easily see which points correspond,
# for example by highlighting the same x value in both subplots when hovering a point in one subplot

# Generate foot down/up

# Present results

np.save(filename + '_quite_nice', tracks)

print('Done and done.')
