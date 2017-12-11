import cv2
import numpy as np
import os


def mouse_callback(event, x, y, flags, pos_anno):
    left_button_is_down = flags & cv2.EVENT_FLAG_LBUTTON
    if event == cv2.EVENT_LBUTTONDOWN or left_button_is_down:
        if marked != 0:
            pos_anno[frame, marked - 1] = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if marked != 0:
            pos_anno[frame, marked - 1] = (np.nan, np.nan)


def interpolate_keypoints(pos_anno):
    nframes, nblobs, npos = pos_anno.shape
    for blob in range(nblobs):
        for pos in range(npos):
            missing_frames = np.where(np.isnan(pos_anno[:, blob, pos]))[0]
            existing_frames = np.where(~np.isnan(pos_anno[:, blob, pos]))[0]
            if existing_frames.size:
                pos_anno[missing_frames, blob, pos] = np.interp(missing_frames,
                                                                existing_frames,
                                                                pos_anno[existing_frames, blob, pos],
                                                                left=np.nan, right=np.nan)


if __name__ != "__main__":
    exit(0)

filename = 'rolf_markerless/rolf_markerless_%04d.jpg'

if '%04d' in filename:
    video_name = os.path.split(filename)[1].split('_%04d')[0]
    directory = 'input-images/'
else:
    video_name = os.path.splitext(filename)[0]
    directory = 'input-videos/'

cap = cv2.VideoCapture(directory + filename)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame = 0
marked = 0
try:
    pos_anno = np.load('annotations/' + video_name + '-positions.npy')
except IOError:
    pos_anno = np.zeros(shape=[nframes, 4, 2], dtype=np.uint16) * np.nan

w = cv2.namedWindow('Position Annotator', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Position Annotator', mouse_callback, pos_anno)
cv2.resizeWindow('Position Annotator', width, height)
font = cv2.FONT_HERSHEY_TRIPLEX
notations = ['empty', 'Left Toe', 'Left Heel', 'Right Toe', 'Right Heel']
blob_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

loop = True
cap.set(1, frame)
ret, clean_im = cap.read()
while loop:
    im = np.copy(clean_im)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Reset r', (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Save S (shift + s)', (0, 70), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Mark: Mouseclick (set with left, remove with right)', (0, 85), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Interpolate: !', (0, 100), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Change markers: i,k,o,l,p', (0, 115), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Frame:' + str(frame), (0, 130), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if marked != 0:
        cv2.putText(im, notations[marked], (0, height - 40), fontFace=font, fontScale=3,
                    color=blob_color[marked - 1])
        cv2.putText(im, "Last clicked position: {}, {}".format(*pos_anno[frame, marked - 1]), (0, height - 160),
                    fontFace=font, fontScale=0.5, color=(0, 0, 255))
    for blob in range(4):
        if not np.any(np.isnan(pos_anno[frame, blob])):  # Are you a nun?
            cv2.drawMarker(im, tuple(pos_anno[frame, blob].astype(np.uint16)), thickness=2, markerSize=10,
                           markerType=cv2.MARKER_TILTED_CROSS, color=blob_color[blob])
    cv2.imshow('Position Annotator', im)
    while True:
        pressed_key = cv2.waitKey(50) & 0xFF
        if pressed_key == 255:  # no key return
            break
        if pressed_key == ord('d'):
            if frame < nframes-1:
                frame = frame + 1
                ret, clean_im = cap.read()
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
                cap.set(1, frame)
                ret, clean_im = cap.read()
        elif pressed_key == ord('q'):
            loop = False
        elif pressed_key == ord('e'):
            frame = 0
            cap.set(1, frame)
            ret, clean_im = cap.read()
        elif pressed_key == ord('r'):
            pos_anno[:, :, :] = np.nan
        elif pressed_key == ord('i'):
            marked = 1
        elif pressed_key == ord('k'):
            marked = 2
        elif pressed_key == ord('o'):
            marked = 3
        elif pressed_key == ord('l'):
            marked = 4
        elif pressed_key == ord('p'):
            marked = 0
        elif pressed_key == ord('!'):
            interpolate_keypoints(pos_anno)
        elif pressed_key == ord('S'):
            np.save('annotations/' + video_name + '-positions', pos_anno)
        else:
            continue
        break

cv2.destroyAllWindows()

"""
p_anno = np.load('annotations/' + video_name + '-positions.npy')

w = cv2.namedWindow('Position Check', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Position Check', width, height)
font = cv2.FONT_HERSHEY_TRIPLEX
loop = True
cap.set(1, frame)
ret, clean_im = cap.read()
while loop:
    im = np.copy(clean_im)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Frame:' + str(frame), (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))

    for blob in range(4):
        if not np.any(np.isnan(p_anno[frame, blob])):
            cv2.drawMarker(im, tuple(pos_anno[frame, blob].astype(np.uint16)), thickness=2, markerSize=10,
                           markerType=cv2.MARKER_TILTED_CROSS, color=blob_color[blob])
    cv2.imshow('Position Check', im)
    while True:
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes-1:
                frame = frame + 1
                ret, clean_im = cap.read()
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
                cap.set(1, frame)
                ret, clean_im = cap.read()
        elif pressed_key == ord('q'):
            loop = False
        elif pressed_key == ord('e'):
            frame = 0
            cap.set(1, frame)
            ret, clean_im = cap.read()
        else:
            continue
        break

cv2.destroyAllWindows()
"""