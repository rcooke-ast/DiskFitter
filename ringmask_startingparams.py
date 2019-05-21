"""
Using the output from annulus fits with the kinmspy_ringmask.py code,
this code sets the starting parameters for the chi-sq.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d

outvals = np.load("outvals.npy")
outerrs = np.load("outerrs.npy")
rings = np.linspace(0.0, 3.0, 20)
rad = np.linspace(rings[0], rings[-1], 1000)

parvals = [[0.0, 0.15789474, 0.31578947, 0.47368421, 0.63157895, 0.78947368,
 0.94736842, 1.10526316, 1.26315789, 1.57894737,
 1.89473684, 2.21052632, 2.52631579, 2.84210526],  # SB prof
           [0.0, 0.15, 0.3, 0.5, 0.8, 1.15, 1.6, 2.0, 3.0],  # posang
           [0.0, 0.25, 0.50, 1.0, 2.0, 3.0],  # inc
           0.0,  # xcen
           0.0,  # ycen
           0.1,  # vcen
           0.8,  # Mass
           [0.0, 0.15, 0.3, 0.5, 0.650, 1.0, 2.0, 3.0]]  # gas sigma

inpvals = [[454.12234286, 393.36149741, 286.80762835, 216.90814331, 207.93973294,
 161.50987764, 140.72118948, 117.95176048,  98.90525955,
  68.81306633,  48.03526046, 32.32684097, 21.07468516,  14.83180197],  # SB prof
           [172, 160.0, 154.8, 155.4, 152.7, 153.3, 152.5, 152.2, 151.5],  # posang
           [7.35403098, 7.63412163, 7.70179149, 7.35666027, 5.84205372, 5.11887352],  # inc
           -0.01216,  # xcen
           -0.02932,  # ycen
           0.07400,  # vcen
           0.84,  # Mass
           [0.078, 0.078, 0.085, 0.13, 0.223, 0.22, 0.17, 0.14]]  # gas sigma




for ii in range(outvals.shape[0]):
    # Plot the fitted data
    plt.subplot(3, 3, ii + 1)
    plt.plot(rings[:-1], outvals[ii, :], marker='o', color='b')
    plt.errorbar(rings[:-1], outvals[ii, :], outerrs[ii, :], fmt='none', color='b')
    if ii == outvals.shape[0]-1:
        break
    # Interpolate/Estimate new starting params
    if type(parvals[ii]) is float:
        xmod = rings
        ymod = parvals[ii]*np.ones(xmod.size)
        print(np.mean(outvals[ii, :]))
    else:
        if ii == 0:
            print(outvals[ii, :])
            xmod, ymod = 0.0, 0.0
        else:
            spl = UnivariateSpline(rings[:-1], outvals[ii, :], bbox=[parvals[ii][0], parvals[ii][-1]], ext='extrapolate', k=3, s=5)
            xmod = parvals[ii]
            ymod = spl(xmod)
            print(ymod)
    # Plot new starting params
    plt.plot(xmod, ymod, 'g--')
    if type(parvals[ii]) is float:
        plt.plot(parvals[ii], inpvals[ii], 'r-')
    else:
        fspl = interp1d(parvals[ii], inpvals[ii], kind='linear', bounds_error=False)
        pspl = fspl(rad)
        plt.plot(rad, pspl, 'r-')
    # plt.errorbar(rings[:-1], outvals[ii, :], outerrs[ii, :], fmt='')
plt.show()

