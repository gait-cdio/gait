from collections import namedtuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ioff()
# import PyOpenPose

def print_keypoint_positions(keypointList):
    i = 1
    for keyPoint in keypointList:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        # print("KeyPoint nr:" + str(i) + "x:" + str(x) + "y:" + str(y) )
        i = i + 1
    print("Number of blobs =", len(keypointList))

def extract_median_circle(img, xpos, ypos, radius):
    cir=np.zeros(img.shape,np.uint8)
    cv2.circle(cir,center=(xpos,ypos),radius=radius,color=255,thickness=-1)
    return np.median(img[(cir == 255)])

# Low scores if similar
def feature_distance(PointFeature1, PointFeature2, distanceweight=1, sizeweight=0, hueweight=0, timeweight=0):
    x1, y1 = PointFeature1.position
    x2, y2 = PointFeature2.position
    d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    s2 = (PointFeature1.size - PointFeature2.size) ** 2
    h = np.abs(PointFeature1.hue-PointFeature2.hue)
    t = np.abs(PointFeature1.frame-PointFeature2.frame)
    s = np.sqrt(s2)
    d = np.sqrt(d2)
    return d * distanceweight + s * sizeweight + h * hueweight + t * timeweight

clicked_position = None

def clickForColor(event, x, y, flags, param):
    global clicked_position
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Clicked ', x, y)
        clicked_position = (x, y)



PointFeatures = namedtuple('PointFeatures', ['position', 'size', 'hue', 'frame'])

videoname = 'pinkdot'
cap = cv2.VideoCapture(videoname + '.mp4')

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(videoname +'.avi', fourcc, fps, (width, height))
counter = 0
pointbuffer = []

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.minThreshold = 10
blob_params.maxThreshold = 180
blob_params.thresholdStep = 20
blob_params.minRepeatability = 1
blob_params.filterByCircularity = False
blob_params.minCircularity = 0.5
blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.3
blob_params.minDistBetweenBlobs = 500
blob_params.filterByArea = True
blob_params.minArea = 60
blob_params.maxArea = 50000
blob_params.filterByConvexity = True
blob_params.minConvexity = 0.9
blob_params.filterByColor = 0
blob_params.blobColor = 100

detector = cv2.SimpleBlobDetector_create(blob_params)

cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Keypoints', 800,600)
cv2.setMouseCallback('Keypoints',clickForColor)
ret,img=cap.read()

roi_rect = cv2.selectROI('Select color', img)
x, y, w, h = roi_rect
roi = img[y:(y+h),x:(x+w)]
cv2.destroyWindow('Select color')

roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

goal_hsv = np.mean(roi_hsv, axis=(0, 1))
hsv_variance = np.var(roi_hsv, axis=(0, 1))
out_of_bounds = False

if hsv_variance[0] < 1000:
    hist, bin_edges = np.histogram(roi_hsv[:, :, 0],bins = 10,range=[0,180])
    ind = np.argpartition(hist,-3)[-3:]
    bin_starts = bin_edges[ind]
    bin_ends = bin_edges[ind + 1]
    out_of_bounds = True


#huehue=extractMedianCircle(img[:,:,1],150,150,50)
#print('test1: ',huehue)
#print('test2: ',img[150,150,1])


paused = False

t_coords = []
x_coords = []
y_coords = []

while (cap.isOpened()):

    ret, img = cap.read()
    if ret == True:
        #img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        # h,s gets lot of sample nose at low/high intensities
        # vthresh = 40
        # ret2, th2 = cv2.threshold(v, vthresh, 255, cv2.THRESH_BINARY)
        # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')
        # ret2, th2 = cv2.threshold(v, 255 - vthresh, 255, cv2.THRESH_BINARY_INV)
        # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')

        lower_bound = np.array(goal_hsv - np.sqrt(hsv_variance))
        upper_bound = np.array(np.clip(goal_hsv + np.sqrt(hsv_variance),[0,0,0],[180,255,255]))

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        if out_of_bounds:
            mask = np.zeros(np.shape(hsv[:,:,0]),dtype='uint8')
            for ind in range(0,len(bin_starts)):
                lower_bound = np.array([bin_starts[ind],50,50])
                upper_bound = np.array([bin_ends[ind],255,255])

                temp = cv2.inRange(hsv, lower_bound, upper_bound)
                #cv2.imshow("Keypoints", temp)
                #cv2.waitKey(0)
                mask = cv2.bitwise_or(temp, mask)
                #cv2.imshow("Keypoints", mask)
                #cv2.waitKey(0)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
 
        filtered = cv2.GaussianBlur(res, (15, 15), 1)
        keypoints = detector.detect(filtered)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(filtered, keypoints, np.array([]), (0, 255, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        pointfeatures = list(map(lambda keypoint: PointFeatures(
            position=keypoint.pt,
            size=keypoint.size,
            hue=float(h[int(keypoint.pt[1]), int(keypoint.pt[0])]),
            frame=counter
        ), keypoints))

        pointbuffer.append(pointfeatures)
        # print(type(keybuffer[0][0]))

        num_keypoints = len(keypoints)
        #print("Detected keypoints (before filtering):", num_keypoints)

        keypoints = list(filter(lambda keypoint: keypoint.pt[0] < 1600, keypoints))

        num_keypoints = len(keypoints)
        #print("Detected keypoints (after filtering):", num_keypoints)

        if num_keypoints > 0:
            x_coords.append(keypoints[num_keypoints - 1].pt[0])
            y_coords.append(keypoints[num_keypoints - 1].pt[1])
            t_coords.append(counter)

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

# plt.plot(x_coords, y_coords)
# plt.ylim(max(y_coords), min(y_coords))  # Reverse y axis because we have image coordinates
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
# plt.plot(t_coords, x_coords, t_coords, y_coords)
# plt.xlabel('t')
# plt.legend()
# plt.show()

pointString=[]
similarity_threshold=100


# Ugly pointmatching code...  I am sorry ;(
# It is done frame by frame more or less.
for frame_index in range(0, len(pointbuffer)):
    # Current frame
    pNext = pointbuffer[frame_index]
    # Simularities between current frame and previous pointstrings


    num_previous_points = len(pointString)
    num_detections = len(pNext)
    if num_detections == 0:
        pass
    # First detection of any points
    elif num_previous_points == 0:
        for point_index in range(0, num_detections):
            newPointList = [pNext[point_index]]
            pointString.append(newPointList)

    # Comparing previous strings and current frame
    else:
        simMat = np.zeros((num_previous_points, num_detections))
        for ix in range(0, num_previous_points):
            for point_index in range(0, num_detections):
                # Only distance at the moment ( 0,0,0 )
                simMat[ix, point_index]=feature_distance(pointString[ix][-1], pNext[point_index], sizeweight=0, hueweight=4, timeweight=1)

        while True:
            try:
                best_match_position = np.unravel_index(np.nanargmin(simMat), (num_previous_points, num_detections))
            except ValueError:
                break
            previous_point_index, new_point_index = best_match_position

            if simMat[best_match_position] < similarity_threshold:
                pointString[previous_point_index].append(pNext[new_point_index])
            else:
                newPointString = [pNext[new_point_index]]
                pointString.append(newPointString)

            simMat[:, new_point_index] = np.nan
            simMat[previous_point_index, :] = np.nan

extract_position_square = lambda item: item.position[0] ** 2 + item.position[1] ** 2
position_variances = [np.var(list(map(extract_position_square, s))) for s in pointString]
plt.plot(np.log(position_variances), 'o')
plt.show()

#pointString = [s for s in pointString if len(s) > 10]

f, axarr = plt.subplots(2, sharex=True)

for index in range(0, len(pointString)):
    curve = pointString[index]
    t_c=[p.frame for p in curve]
    x_c=[p.position[0] for p in curve]
    y_c=[p.position[1] for p in curve]
    xline = axarr[0].plot(t_c, x_c, 'o-', markersize=2, label='point {}'.format(index))
    yline = axarr[1].plot(t_c, y_c, 'o-', markersize=2, label='point {}'.format(index))
plt.xlabel('frame')
#axarr[0].ylabel('x')
#axarr[1].ylabel('y')
plt.legend()
plt.show()


print('You did the thing :)')

