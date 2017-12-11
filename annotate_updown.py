import numpy as np
import cv2
import os

import utils

# filename = 'john_markerless/john_markerless_%04d.jpg'
filename = '4farger.mp4'

if '%04d' in filename:
    video_name = os.path.split(filename)[1].split('_%04d')[0]
else:
    video_name = os.path.splitext(filename)[0]

cap = cv2.VideoCapture('input-videos/' + filename)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('Gait Annotator', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gait Annotator', width, height)

try:
    annotations = np.load('annotations/' + video_name + '-up_down.npy')
    if len(annotations.shape) == 1:
        annotations = utils.annotationToUpDown(annotations)
except FileNotFoundError:
    annotations = np.zeros(shape=(2, nframes), dtype='uint8') * np.nan

font = cv2.FONT_HERSHEY_TRIPLEX
frame = 0
notations = ['emp', 'Left Up', 'Left Down', 'Right Up', 'Right Down']

loop = True
cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
ret, clean_im = cap.read()
while loop:
    im = np.copy(clean_im)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Reset r', (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Save S (shift + s)', (0, 70), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Mark: i,k,o,l,p', (0, 85), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Frame: ' + str(frame), (0, 100), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if not np.isnan(annotations[0, frame]):
        if annotations[0, frame]:
            text = 'Left Up'
            ypos = height - 140
        else:
            text = 'Left Down'
            ypos = height - 40
        cv2.putText(im, text, (0, ypos), fontFace=font, fontScale=2, color=(0, 0, 255))
    if not np.isnan(annotations[1, frame]):
        if annotations[1, frame]:
            text = 'Right Up'
            ypos = height - 140
        else:
            text = 'Right Down'
            ypos = height - 40
        cv2.putText(im, text, (width - 500, ypos), fontFace=font, fontScale=2, color=(0, 0, 255))
    cv2.imshow('Gait Annotator', im)
    while True:
        pressed_key = cv2.waitKey(50) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes - 1:
                frame = frame + 1
                ret, clean_im = cap.read()

                if np.isnan(annotations[0, frame]):
                    annotations[0, frame] = annotations[0, frame - 1]
                if np.isnan(annotations[1, frame]):
                    annotations[1, frame] = annotations[1, frame - 1]
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, clean_im = cap.read()
        elif pressed_key == ord('q'):
            loop = False
        elif pressed_key == ord('e'):
            frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, clean_im = cap.read()
        elif pressed_key == ord('r'):
            annotations = np.zeros(shape=nframes, dtype='uint8')
        elif pressed_key == ord('i'):
            if annotations[0, frame] == 1:
                annotations[0, frame] = np.nan
            else:
                annotations[0, frame] = 1
        elif pressed_key == ord('k'):
            if annotations[0, frame] == 0:
                annotations[0, frame] = np.nan
            else:
                annotations[0, frame] = 0
        elif pressed_key == ord('o'):
            if annotations[1, frame] == 1:
                annotations[1, frame] = np.nan
            else:
                annotations[1, frame] = 1
        elif pressed_key == ord('l'):
            if annotations[1, frame] == 0:
                annotations[1, frame] = np.nan
            else:
                annotations[1, frame] = 0
        elif pressed_key == ord('p'):
            annotations[:, frame] = np.nan
        elif pressed_key == ord('S'):
            np.save('annotations/' + video_name + '-up_down', annotations)
        else:
            continue
        break
cv2.destroyAllWindows()
