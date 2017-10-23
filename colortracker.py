import numpy as np
import cv2
# import PyOpenPose


def print_keypoint_positions(keypointList):
    i = 1
    for keyPoint in keypointList:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        # print("KeyPoint nr:" + str(i) + "x:" + str(x) + "y:" + str(y) )
        i = i + 1
    print("Number of blobs =", len(keypointList))


cap = cv2.VideoCapture('4farger.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('4farger.avi', fourcc, fps, (width, height))
counter = 0
keybuffer = np.array([])

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.minThreshold = 10
blob_params.maxThreshold = 180
blob_params.thresholdStep = 30
blob_params.minRepeatability = 1
blob_params.filterByCircularity = False
blob_params.minCircularity = 0.5
blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.3
blob_params.minDistBetweenBlobs = 500
blob_params.filterByArea = True
blob_params.minArea = 60
blob_params.maxArea = 200
blob_params.filterByConvexity = True
blob_params.minConvexity = 0.9
blob_params.filterByColor = 0
blob_params.blobColor = 100

detector = cv2.SimpleBlobDetector_create(blob_params)

cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Keypoints', 800,600)

paused = False

while (cap.isOpened()):

    ret, img = cap.read()
    if ret == True:
        #img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # h, s, v = cv2.split(hsv)
        # h,s gets lot of sample nose at low/high intensities
        # vthresh = 40
        # ret2, th2 = cv2.threshold(v, vthresh, 255, cv2.THRESH_BINARY)
        # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')
        # ret2, th2 = cv2.threshold(v, 255 - vthresh, 255, cv2.THRESH_BINARY_INV)
        # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')

        # define range of blue color in HSV
        lower_pink = np.array([150, 50, 50])
        upper_pink = np.array([180, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)

 
        filtered = cv2.GaussianBlur(res, (15, 15), 1)
        keypoints = detector.detect(filtered)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(filtered, keypoints, np.array([]), (0, 255, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keybuffer = np.append(keybuffer, np.array(keypoints))
        # print(type(keybuffer[0][0]))
        print("Detected keypoints:", len(keypoints))
        print_keypoint_positions(keybuffer)
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        out.write(im_with_keypoints)

        if paused:
            delay = 0
        else:
            delay = 1

        pressed_key = cv2.waitKey(delay) & 0xFF
        if pressed_key == ord(' '):
            paused = not paused
        elif pressed_key == ord('q'):
            break
    else:
        break
    counter += 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print('You did the thing :)')


# Low scores if similar
def simularityDistance(keypoint1, keypoint2, sizeweight):
    x1, y1 = keypoint1.pt
    x2, y2 = keypoint2.pt
    d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    s2 = (keypoint1.size - keypoint2.size) ** 2
    return d2 + s2 * sizeweight


def simularityMatrix(keypoints1, keypoints2):
    keypoints1.repmat()