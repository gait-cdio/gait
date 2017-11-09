import numpy as np
import cv2
import imageio
import os

videoname='twofeet.mp4'
reader = imageio.get_reader(videoname)
imsize=reader.get_meta_data()['size']
nframes = reader.get_meta_data()['nframes']
fps = reader.get_meta_data()['fps']

cv2.namedWindow('Gait Annotator',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gait Annotator', imsize[0],imsize[1])
font = cv2.FONT_HERSHEY_TRIPLEX
frame=0
annotations = np.zeros(shape=nframes,dtype='uint8')
notations = ['emp','Left Up', 'Left Down', 'Right Up', 'Right Down']
loop = True
while loop:
    imrgb = reader.get_data(frame)
    im = cv2.cvtColor(imrgb, cv2.COLOR_RGB2BGR)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Reset r', (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Save s', (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Mark: i,k,o,l,p', (0, 70), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if annotations[frame] != 0:
        cv2.putText(im, notations[annotations[frame]], (0,imsize[1]-40), fontFace=font, fontScale=3,color=(0,0,255))
    cv2.imshow('Gait Annotator',im)
    while True:
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes-1:
                frame = frame + 1
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
        elif pressed_key == ord('q'):
            loop=False
        elif pressed_key == ord('e'):
            frame=0
        elif pressed_key == ord('r'):
            annotations=np.zeros(shape=nframes,dtype='uint8')
        elif pressed_key == ord('i'):
            annotations[frame]= 1
        elif pressed_key == ord('k'):
            annotations[frame]= 2
        elif pressed_key == ord('o'):
            annotations[frame]= 3
        elif pressed_key == ord('l'):
            annotations[frame]= 4
        elif pressed_key == ord('p'):
            annotations[frame]= 0
        elif pressed_key == ord('s'):
            np.save(os.path.splitext(videoname)[0],annotations)
        else:
            continue
        break
cv2.destroyAllWindows()

anno = np.load(os.path.splitext(videoname)[0] + ".npy")

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

anno = annotationToOneHot(anno)



cv2.namedWindow('Check',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Check', imsize[0],imsize[1])
ballsize=100
loop = True
while loop:
    imrgb = reader.get_data(frame)
    im = cv2.cvtColor(imrgb, cv2.COLOR_RGB2BGR)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if anno[0,frame] == 1:
        cv2.circle(im, (ballsize,imsize[1]-ballsize),ballsize,thickness=-1,color=(0,0,255))
    if anno[1,frame] == 1:
        cv2.circle(im, (imsize[0]-ballsize, imsize[1]-ballsize), ballsize, thickness=-1, color=(0, 255, 0))
    cv2.imshow('Check',im)
    while True:
        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes-1:
                frame = frame + 1
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
        elif pressed_key == ord('q'):
            frame= loop=False
        elif pressed_key == ord('e'):
            frame=0
        else:
            continue
        break
cv2.destroyAllWindows()

frame=0
marked=0
pos_anno=np.zeros(shape=[4,nframes],dtype=(np.uint16,2))


def mouse_callback(event, x, y, flags, pos_anno):
    if event == cv2.EVENT_LBUTTONDOWN:
        if marked != 0:
            pos_anno[marked-1,frame]=(x,y)
            print(x,y)



w=cv2.namedWindow('Position Annotator',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Position Annotator', imsize[0],imsize[1])
font = cv2.FONT_HERSHEY_TRIPLEX
notations = ['emp','Left Up', 'Left Down', 'Right Up', 'Right Down']
cv2.setMouseCallback('Position Annotator', mouse_callback, pos_anno)

loop = True
while loop:

    imrgb = reader.get_data(frame)
    im = cv2.cvtColor(imrgb, cv2.COLOR_RGB2BGR)
    cv2.putText(im, 'Step: a,d', (0, 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'To start: e', (0, 40), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Quit: q', (0, 25), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Reset r', (0, 55), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Save s', (0, 70), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    cv2.putText(im, 'Mark: Mouseclick', (0, 85), fontFace=font, fontScale=0.5, color=(0, 0, 255))
    if marked != 0:
        cv2.putText(im, notations[marked], (0, imsize[1] - 40), fontFace=font, fontScale=3,
                    color=(0, 0, 255))
        for blob in range(4):
            if not np.all(pos_anno[blob,frame] == 0): # all zeros
                cv2.circle(im, tuple(pos_anno[blob,frame]), 10, thickness=-1, color=(blob*50, 255, 0))
    cv2.imshow('Position Annotator',im)
    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('d'):
            if frame < nframes-1:
                frame = frame + 1
        elif pressed_key == ord('a'):
            if frame > 0:
                frame = frame - 1
        elif pressed_key == ord('q'):
            loop=False
        elif pressed_key == ord('e'):
            frame=0
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
            pos_anno[:,frame,:]=0
        elif pressed_key == ord('s'):
            np.save(os.path.splitext(videoname)[0] + '-positions',annotations)
        else:
            continue
        break

cv2.destroyAllWindows()
