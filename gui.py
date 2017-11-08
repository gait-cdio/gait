import cv2
import numpy as np

#TODO(John):
# Möjligör flera olika trösklar
# Bättre utnyttjade av klicket i bilden


class WindowGUI: #Klass som innehåller information för GUI:t
    def __init__(self, video_reader):
        self.windowName = 'Choose HSV threshold'
        self.maskedWindow = 'MaskedIm'
        self.lowerBound = np.array([140, 100, 100])
        self.higherBound = np.array([180, 255, 255])

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 500, 500)

        self.video_reader = video_reader
        self.read_image(frame=0)
        self.mask = np.zeros(np.size(self.bgrIm))

    def read_image(self, frame):
        img = cv2.cvtColor(self.video_reader.get_data(frame), cv2.COLOR_RGB2BGR)
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


def set_threshold(video_reader):
    w = WindowGUI(video_reader)

    cv2.createTrackbar('Hue lower threshold', w.windowName,  140, 180, w.hlowCallback)
    cv2.createTrackbar('Hue higher threshold', w.windowName, 180, 180, w.hhighCallback)

    cv2.createTrackbar('Saturation lower threshold', w.windowName, 100, 256, w.slowCallback)
    cv2.createTrackbar('Saturation higher threshold', w.windowName, 256, 256, w.shighCallback)

    cv2.createTrackbar('Value lower threshold', w.windowName, 100, 256, w.vlowCallback)
    cv2.createTrackbar('Value higher threshold', w.windowName, 256, 256, w.vhighCallback)
    cv2.createTrackbar('Frame', w.windowName, 0, len(video_reader) - 1, w.read_image)
    cv2.setMouseCallback(w.windowName, mouseCallback, w)

    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            cv2.destroyWindow(w.windowName)
            cv2.destroyWindow(w.maskedWindow)
            break

    return w.lowerBound, w.higherBound
