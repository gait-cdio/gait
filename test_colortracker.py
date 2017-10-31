import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

from collections import namedtuple

from colortracker import match_points, PointFeatures

plt.ioff()
# import PyOpenPose

ColorInterval = namedtuple('ColorInterval', ['mean_hsv', 'variance'])

videoname = 'onefoot'
cache_filename = videoname + '.detections.npy'

if 'cached' in sys.argv and os.path.isfile(cache_filename):
    pointbuffer = np.load(cache_filename)
else:
    cap = cv2.VideoCapture(videoname + '.mp4')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    blob_params.minDistBetweenBlobs = 10
    blob_params.filterByArea = True
    blob_params.minArea = 60
    blob_params.maxArea = 50000
    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.9
    blob_params.filterByColor = 0
    blob_params.blobColor = 100

    detector = cv2.SimpleBlobDetector_create(blob_params)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    ret,img=cap.read()

    cv2.namedWindow('Select color',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select color', 1000,1000)
    roi_rects = cv2.selectROIs('Select color', img)

    rois_hsv = []
    color_intervals = []

    for roi_rect in roi_rects:
        x, y, w, h = roi_rect
        roi = img[y:(y+h),x:(x+w)]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        goal_hsv = np.mean(roi_hsv, axis=(0, 1))
        hsv_variance = np.var(roi_hsv, axis=(0, 1))

        histl, binl = np.histogram(roi_hsv[:, :, 0])
        hista, bina = np.histogram(roi_hsv[:, :, 1])
        histb, binb = np.histogram(roi_hsv[:, :, 2])
        figHandle, axisHandle = plt.subplots(ncols = 3)
        axisHandle[0].plot(histl)
        axisHandle[1].plot(hista)
        axisHandle[2].plot(histb)
        plt.show()

        hsv_variance[0] *= 2

        color_intervals.append(ColorInterval(mean_hsv=goal_hsv, variance=hsv_variance))

    cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Keypoints', 800,600)

    cv2.destroyWindow('Select color')

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_of_bounds = False

   # if hsv_variance[0] > 1000:
   #     hist, bin_edges = np.histogram(roi_hsv[:, :, 0],bins = 10,range=[0,180])
   #     ind = np.argpartition(hist,-2)[-2:]
   #     bin_starts = bin_edges[ind]
   #     bin_ends = bin_edges[ind + 1]
   #     out_of_bounds = True


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
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #
            # h, s, v = cv2.split(hsv)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)

            # h,s gets lot of sample nose at low/high intensities
            # vthresh = 40
            # ret2, th2 = cv2.threshold(v, vthresh, 255, cv2.THRESH_BINARY)
            # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')
            # ret2, th2 = cv2.threshold(v, 255 - vthresh, 255, cv2.THRESH_BINARY_INV)
            # h = np.multiply(h, (th2 / 255).astype(np.uint8), dtype='uint8')

            #lower_bound = np.array(goal_hsv - np.sqrt(hsv_variance))
            #upper_bound = np.array(np.clip(goal_hsv + np.sqrt(hsv_variance),[0,0,0],[180,255,255]))

            #mask = cv2.inRange(hsv, lower_bound, upper_bound)
            #if out_of_bounds:
            mask = np.zeros(np.shape(hsv[:,:,0]),dtype='uint8')
            for ind in range(0,len(color_intervals)):
                #lower_bound = np.array([bin_starts[ind],50,50])
                #upper_bound = np.array([bin_ends[ind],255,255])

                lower_bound = color_intervals[ind].mean_hsv - np.sqrt(color_intervals[ind].variance)
                upper_bound = color_intervals[ind].mean_hsv + np.sqrt(color_intervals[ind].variance)

                temp = cv2.inRange(hsv, lower_bound, upper_bound)
                #cv2.imshow("Keypoints", temp)
                #cv2.waitKey(0)
                mask = cv2.bitwise_or(temp, mask)
                    #cv2.imshow("Keypoints", mask)
                    #cv2.waitKey(0)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(img, img, mask=mask)

            filtered = cv2.GaussianBlur(res, (15, 15), 1)
            keypoints = detector.detect(res)
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the qcircle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0, 255, 0),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            pointfeatures = list(map(lambda keypoint: PointFeatures(
                position=keypoint.pt,
                size=keypoint.size,
                hue=float(h[int(keypoint.pt[1]), int(keypoint.pt[0])]),
                frame=counter
            ), keypoints))

            pointbuffer.append(pointfeatures)
            # print(type(keybuffer[0][0]))

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

    np.save(cache_filename, pointbuffer)

pointString=[]
similarity_threshold=100

for frame_index in range(0, len(pointbuffer)):
    # Current frame
    new_points = pointbuffer[frame_index]

    match_points(new_points, pointString, similarity_threshold)


def extract_position_square(item):
    return item.position[0] ** 2 + item.position[1] ** 2


# position_variances = [np.var(list(map(extract_position_square, s))) for s in pointString]
# plt.plot(np.log(position_variances), 'o')
# plt.show()

#pointString = [s for s in pointString if len(s) > 10]

f, axes = plt.subplots(ncols=2)

for index in range(0, len(pointString)):
    curve = pointString[index]
    t_c=[p.frame for p in curve]
    x_c=[p.position[0] for p in curve]
    y_c=[p.position[1] for p in curve]
    xline = axes[0].plot(t_c, x_c, 'o-', markersize=2)
    yline = axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()


print('You did the thing :)')

