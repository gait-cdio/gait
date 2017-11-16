import cv2
import numpy as np
from collections import namedtuple
#TODO(John):
# Möjligör flera olika trösklar
# Bättre utnyttjade av klicket i bilden
# Använda QT istället för openCVs GUI. openCV GUIt ska egentligen inte användas till mer än debugging. Därför finns ingen support för knappar o.s.v.

#markerThreshold = namedtuple('markerThreshold', 'jointName lowerBound upperBound') # Tänker mig en sån här grej för att spara värden

class WindowGUI: #Klass som innehåller information för GUI:t
    def __init__(self, cap, init_thresholds=None):
        self.windowName = 'Choose HSV threshold'
        self.maskedWindow = 'MaskedIm'

        self.multipleBounds = False

        if(init_thresholds):
            self.lowerBound = init_thresholds[0]
            self.higherBound = init_thresholds[1]
        else:
            self.lowerBound = np.array([140, 100, 100])
            self.higherBound = np.array([180, 255, 255])

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 500, 500)

        self.cap = cap
        self.read_image(frame=0)
        self.mask = np.zeros(np.size(self.bgrIm))

    def read_image(self, frame):
        self.cap.set(1, frame)
        ret, img = self.cap.read()
        self.bgrIm = img
        self.hsvIm = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.updateWindowImage()

    # Anropsfunktioner för trackbars
    def hlowCallback(self, pos):
        self.lowerBound[0] = pos
        self.updateWindowImage()

    def slowCallback(self, pos):
        self.lowerBound[1] = pos
        self.updateWindowImage()

    def vlowCallback(self, pos):
        self.lowerBound[2] = pos
        self.updateWindowImage()

    def hhighCallback(self, pos):
        self.higherBound[0] = pos
        self.updateWindowImage()

    def shighCallback(self, pos):
        self.higherBound[1] = pos
        self.updateWindowImage()

    def vhighCallback(self, pos):
        self.higherBound[2] = pos
        self.updateWindowImage()

    # Uppdatering för masken
    def updateWindowImage(self):
        self.mask = cv2.inRange(self.hsvIm, self.lowerBound, self.higherBound)

        cv2.imshow(self.windowName, self.bgrIm / 255 + (self.mask[:, :, np.newaxis] / 255))

    # Uppdatering för trackbarsen när de sätts genom att klicka i bilden
    def updateTrackbars(self):
        cv2.setTrackbarPos('Hue lower threshold', self.windowName, self.lowerBound[0])
        cv2.setTrackbarPos('Hue higher threshold', self.windowName, self.higherBound[0])
        cv2.setTrackbarPos('Saturation lower threshold', self.windowName, self.lowerBound[1])
        cv2.setTrackbarPos('Saturation higher threshold', self.windowName, self.higherBound[1])
        cv2.setTrackbarPos('Value lower threshold', self.windowName, self.lowerBound[2])
        cv2.setTrackbarPos('Value higher threshold', self.windowName, self.higherBound[2])


def mouseCallback(event, x, y, flags, w):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x, y = ' + str((x, y)))
        if x >= 5 and y >= 5:
            floodfill_mask = np.zeros((w.bgrIm.shape[0] + 2, w.bgrIm.shape[1] + 2), dtype=np.uint8)
            hsv = cv2.cvtColor(w.bgrIm, cv2.COLOR_BGR2HSV).astype(np.float32)
            h = hsv[:,:,0].astype(np.float32)

            value_to_write_in_mask = 255
            flags = (value_to_write_in_mask << 8) | cv2.FLOODFILL_MASK_ONLY
            cv2.floodFill(hsv, floodfill_mask, (x, y), 255, loDiff=(10, 20, 10), upDiff=(10, 20, 10), flags=flags)

            actual_mask = floodfill_mask[1:-1, 1:-1]
            interesting_hsv_pixels = hsv[actual_mask == 255]

            if not flags & cv2.EVENT_FLAG_CTRLKEY:
                w.lowerBound = np.min(interesting_hsv_pixels, axis=0) - (10, 25, 25)
                w.higherBound = np.max(interesting_hsv_pixels, axis=0) + (10, 25, 25)

                w.lowerBound = np.clip(w.lowerBound, [0, 0, 0], [180, 255, 255]).astype(int)
                w.higherBound = np.clip(w.higherBound, [0, 0, 0], [180, 255, 255]).astype(int)
                w.updateWindowImage()
                w.updateTrackbars()


def set_threshold(cap, init_thresholds=None):
    if not init_thresholds:
        init_thresholds = (np.array([140, 100, 100]), np.array([180, 255, 255]))

    w = WindowGUI(cap, init_thresholds)
    h_lower = init_thresholds[0][0]; h_upper = init_thresholds[1][0]
    s_lower = init_thresholds[0][1]; s_upper = init_thresholds[1][1]
    v_lower = init_thresholds[0][2]; v_upper = init_thresholds[1][2]

    cv2.createTrackbar('Hue lower threshold', w.windowName,  h_lower, 180, w.hlowCallback)
    cv2.createTrackbar('Hue higher threshold', w.windowName, h_upper, 180, w.hhighCallback)

    cv2.createTrackbar('Saturation lower threshold', w.windowName, s_lower, 256, w.slowCallback)
    cv2.createTrackbar('Saturation higher threshold', w.windowName, s_upper, 256, w.shighCallback)

    cv2.createTrackbar('Value lower threshold', w.windowName, v_lower, 256, w.vlowCallback)
    cv2.createTrackbar('Value higher threshold', w.windowName, v_upper, 256, w.vhighCallback)
    cv2.createTrackbar('Frame', w.windowName, 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, w.read_image)
    cv2.setMouseCallback(w.windowName, mouseCallback, w)

    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            cv2.destroyWindow(w.windowName)
            cv2.destroyWindow(w.maskedWindow)
            break

    return w.lowerBound, w.higherBound

