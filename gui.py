import cv2
import numpy as np

#TODO(John):
# Möjligör flera olika trösklar
# Bättre utnyttjade av klicket i bilden

class WindowGUI: #Klass som innehåller information för GUI:t
    def __init__(self, img):
        self.sourceIm = img
        self.windowIm = img
        self.windowName = 'Choose HSV threshold'
        self.maskedWindow = 'MaskedIm'
        self.lowerBound = np.array([0, 0, 0])
        self.higherBound = np.array([255, 255, 255])
        self.mask = np.ones(np.size(img))

    #Anropsfunktioner för trackbars
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

    #Uppdatering för masken
    def updateWindowImage(self):
        self.mask = cv2.inRange(self.sourceIm, self.lowerBound, self.higherBound)
        self.windowIm = cv2.bitwise_and(self.sourceIm, self.sourceIm, self.mask)
        cv2.imshow(self.maskedWindow, self.mask)

    #Uppdatering för trackbarsen när de sätts genom att klicka i bilden
    def updateTrackbars(self):
        cv2.setTrackbarPos('Hue lower threshold', self.windowName, self.lowerBound[0])
        cv2.setTrackbarPos('Hue higher threshold', self.windowName, self.higherBound[0])
        cv2.setTrackbarPos('Saturation lower threshold', self.windowName, self.lowerBound[1])
        cv2.setTrackbarPos('Saturation higher threshold', self.windowName, self.higherBound[1])
        cv2.setTrackbarPos('Value lower threshold', self.windowName, self.lowerBound[2])
        cv2.setTrackbarPos('Value higher threshold', self.windowName, self.higherBound[2])

def mouseCallback(event, x, y, flags, w):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x: ' +  str(x) + ' y: ' + str(y))
        if (x > 0 and y > 0):
            im = cv2.cvtColor(w.sourceIm, cv2.COLOR_BGR2HSV)
            h,s,v = im[y,x,:]
            w.lowerBound = np.clip(np.array([h-40,s-50,v-50]),[0,0,0],[180,255,255])
            w.higherBound = np.clip(np.array([h+40,s+50,v+50]),[0,0,0],[180,255,255])
            w.updateWindowImage()
            w.updateTrackbars()

#Legacy, trackbars måste ha anropsfunktioner
def nothing():
    pass

def set_threshold(img):
    w = WindowGUI(img)

    cv2.namedWindow(w.windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(w.maskedWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(w.windowName, 500,500)
    cv2.resizeWindow(w.maskedWindow, 800, 500)

    cv2.createTrackbar('Hue lower threshold', w.windowName,  0, 180, w.hlowCallback)
    cv2.createTrackbar('Hue higher threshold', w.windowName, 180, 180, w.hhighCallback)

    cv2.createTrackbar('Saturation lower threshold', w.windowName, 0, 255, w.slowCallback)
    cv2.createTrackbar('Saturation higher threshold', w.windowName, 255, 255, w.shighCallback)

    cv2.createTrackbar('Value lower threshold', w.windowName, 0, 255, w.vlowCallback)
    cv2.createTrackbar('Value higher threshold', w.windowName, 255, 255, w.vhighCallback)
    cv2.imshow(w.windowName, w.windowIm)
    cv2.setMouseCallback(w.windowName, mouseCallback, w)

    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            cv2.destroyWindow(w.windowName)
            cv2.destroyWindow(w.maskedWindow)
            break

    return w.lowerBound, w.higherBound

