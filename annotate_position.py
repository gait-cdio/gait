import cv2
import numpy as np
import os

video_name = '4farger.mp4'
cap = cv2.VideoCapture('input-videos/' + video_name)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('Gait Annotator', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gait Annotator', width, height)

frame = 0
marked = 0
pos_anno = np.zeros(shape=[4, nframes], dtype=(np.uint16, 2))


def mouse_callback(event, x, y, flags, pos_anno):
    if event == cv2.EVENT_LBUTTONDOWN:
        if marked != 0:
            pos_anno[marked - 1, frame] = (x, y)


w = cv2.namedWindow('Position Annotator', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Position Annotator', mouse_callback, pos_anno)
cv2.resizeWindow('Position Annotator', width, height)
font = cv2.FONT_HERSHEY_TRIPLEX
notations = ['emp', 'Left Toe', 'Left Heel', 'Right Toe', 'Right Heel']
blob_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

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
    cv2.putText(im, 'Mark: Mouseclick', (0, 85), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Change markers: i,k,o,l,p', (0, 100), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Frame:' + str(frame), (0, 115), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if marked != 0:
        cv2.putText(im, notations[marked], (0, height - 40), fontFace=font, fontScale=3,
                    color=(0, 0, 255))
        cv2.putText(im, "Last clicked position: {}, {}".format(*pos_anno[marked - 1, frame]), (0, height - 160),
                    fontFace=font, fontScale=0.5, color=(0, 0, 255))
        for blob in range(4):
            if not np.all(pos_anno[blob, frame] == 0):  # all zeros
                cv2.circle(im, tuple(pos_anno[blob, frame]), 10, thickness=-1, color=blob_color[blob])
    cv2.imshow('Position Annotator', im)
    while True:
        pressed_key = cv2.waitKey(50) & 0xFF
        if pressed_key == 255:  # no key return
            break
        if pressed_key == ord('d'):
            if frame < nframes:
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
            pos_anno = np.zeros(shape=[4, nframes, 2], dtype=np.uint8)
        elif pressed_key == ord('i'):
            marked = 1
        elif pressed_key == ord('k'):
            marked = 2
        elif pressed_key == ord('o'):
            marked = 3
        elif pressed_key == ord('l'):
            marked = 4
        elif pressed_key == ord('p'):
            pos_anno[:, frame, :] = 0
        elif pressed_key == ord('S'):
            np.save('annotations/' + os.path.splitext(video_name)[0] + '-positions', pos_anno)
        else:
            continue
        break

cv2.destroyAllWindows()

p_anno = np.load('annotations/' + os.path.splitext(video_name)[0] + '-positions.npy')

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
        if not np.all(p_anno[blob, frame] == 0):  # all zeros
            cv2.circle(im, tuple(p_anno[blob, frame]), 10, thickness=-1, color=blob_color[blob])
    cv2.imshow('Position Check', im)
    while True:
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes:
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
