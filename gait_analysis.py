import cv2
from colortracker import ColorTracker

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
frame_nr = 0
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        detections = keypoint_tracker.detect(img, frame_nr)
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
        break

# Associate keypoints to form tracks
tracks = keypoint_tracker.associate()

# Generate foot down/up

# Present results