import matplotlib.pyplot as plt
import numpy as np
import utils
import scipy.ndimage.filters as filt

appStd = 1.3 # Stddev for applicability function for infill
blurStd = 4 # Stddev for blurring of signal (and derivative)
speedThresh = 0.5 # Multiplier of minimum speed considered to be off the ground compared to max speed

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def curveInFill (curve, indexes):
    newLength = np.max(indexes)-np.min(indexes)+1
    filled = np.zeros(newLength)
    certainty = np.zeros(newLength)
    indOffset = np.min(indexes)

    filled[indexes - indOffset] = curve
    newFilled = np.copy(filled)
    certainty[indexes - indOffset] = 1

    for ii in np.nditer(np.where(certainty == 0)):
        applicability=gaussian(np.arange(0, newLength), mu=ii, sig=appStd)
        newFilled[ii]=np.sum(applicability * certainty*filled)/np.sum(certainty * applicability)

    return newFilled, range(0, newLength) + indOffset

plt.ioff()
#videoname = '4farger'
#vidpoints = videoname + '.detections.npy'
#vidpointsmatched = videoname + '.detections.matched.npy'

#points = np.load(vidpoints)
pointsMatched = np.load('4farger.mp4_quite_nice.npy')
footstates = np.load('4farger.npy')
hot=utils.annotationToOneHot(footstates)
f, axes = plt.subplots(ncols=2)

for index in range(0, len(pointsMatched)):
    curve = pointsMatched[index]
    t_c=[p.frame for p in curve]
    x_c=[p.position[0] for p in curve]
    y_c=[p.position[1] for p in curve]
    fixedx, fixedt = curveInFill(np.array(x_c), np.array(t_c))
    fixedy, fixedt = curveInFill(np.array(y_c), np.array(t_c))
    dx_c = filt.gaussian_filter1d(input=fixedx, sigma=blurStd, order=1) # order=1 lågpass + derivering
    dxline = axes[0].plot(fixedt, -140*dx_c, 'o-', markersize=2)
    thresh=np.where(dx_c<(np.min(dx_c)*speedThresh))
    estStep=np.zeros(dx_c.shape)
    estStep[thresh]=1
    estdxline = axes[0].plot(fixedt, (1+index)*1000 * estStep, 'o-', markersize=2)
    xline = axes[0].plot(fixedt, fixedx, 'o-', markersize=2)
    yline = axes[1].plot(fixedt, fixedy, 'o-', markersize=2)

xleftfoot = axes[0].plot(3000*hot[0,:],'o-',markersize=2) # lägger till ground truth plot

axes[1].invert_yaxis()
plt.show()







