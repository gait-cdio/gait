import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

from collections import namedtuple

from colortracker import match_points, PointFeatures

plt.ioff()
# import PyOpenPose

videoname = '4farger'
cache_filename = videoname + '.detections.npy'

tracker = namedtuple('tracker', ['tracker', 'lower_bound', 'upper_bound', 'name'])

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

    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    ret,img=cap.read()

    cv2.namedWindow('Select color',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select color', 1000,1000)
    roi_rects = cv2.selectROIs('Select color', img)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    rois_hsv = []
    color_intervals = []
    trackers = []

    for roi_rect in roi_rects:
        x, y, w, h = roi_rect
        roi = img[y:(y+h),x:(x+w)]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        goal_hsv = np.mean(roi_hsv, axis=(0, 1))
        hsv_variance = np.var(roi_hsv, axis=(0, 1))
        #
        # histl, binl = np.histogram(roi_hsv[:, :, 0])
        # hista, bina = np.histogram(roi_hsv[:, :, 1])
        # histb, binb = np.histogram(roi_hsv[:, :, 2])
        # figHandle, axisHandle = plt.subplots(ncols = 3)
        # axisHandle[0].plot(histl)
        # axisHandle[1].plot(hista)
        # axisHandle[2].plot(histb)
        # plt.show()
        #
        # hsv_variance[0] *= 2

        lower_bound = goal_hsv - np.sqrt(hsv_variance)
        upper_bound = goal_hsv + np.sqrt(hsv_variance)

        mask = cv2.inRange(hsv, lower_bound,upper_bound)
        thresh_hsv = cv2.bitwise_and(hsv,hsv,mask = mask)

        trackers.append(tracker(tracker=cv2.TrackerMedianFlow_create(),lower_bound=lower_bound, upper_bound=upper_bound, name = str(goal_hsv[0])))
        trackers[-1].tracker.init(thresh_hsv, (x-10, y-10, w+20, h+20)) #kolla för färgtröskling i liten bild, tracka större box.
        cv2.namedWindow(str(goal_hsv[0]), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(goal_hsv[0]), 800, 600)

    cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Keypoints', 800,600)

    cv2.destroyWindow('Select color')

    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
        last_bbox = (x - 10, y - 10, w + 20, h + 20)
        if ret == True:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            for part_tracker in trackers:
                mask = cv2.inRange(hsv,part_tracker.lower_bound,part_tracker.upper_bound)
                thresh_hsv = cv2.bitwise_and(hsv,hsv,mask=mask)

                ok, bbox = part_tracker.tracker.update(thresh_hsv)

                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(thresh_hsv, p1, p2, (255, 0, 0), 2, 1)
                    last_bbox = bbox
                else:
                    cv2.putText(thresh_hsv, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                    part_tracker.tracker.init(thresh_hsv,last_bbox)

            # pointfeatures = list(map(lambda keypoint: PointFeatures(
            #     position=keypoint.pt,
            #     size=keypoint.size,
            #     hue=float(h[int(keypoint.pt[1]), int(keypoint.pt[0])]),
            #     frame=counter
            # ), keypoints))
            #
            pointbuffer.append((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2), counter))
            # # print(type(keybuffer[0][0]))
            #
            # # Show keypoints
            #cv2.imshow(part_tracker.name, mask)
            cv2.imshow('Keypoints',thresh_hsv)
            # out.write(im_with_keypoints)

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


def extract_position_square(item):
    return item.position[0] ** 2 + item.position[1] ** 2


# position_variances = [np.var(list(map(extract_position_square, s))) for s in pointString]
# plt.plot(np.log(position_variances), 'o')
# plt.show()

#pointString = [s for s in pointString if len(s) > 10]

f, axes = plt.subplots(ncols=2)

for index in range(0, len(pointbuffer)):
    curve = pointbuffer[index]
    t_c=curve[2]
    x_c=curve[0]
    y_c=curve[1]
    xline = axes[0].plot(t_c, x_c, 'o-', markersize=2)
    yline = axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()


print('You did the thing :)')

