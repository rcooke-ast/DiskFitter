import os
import pdb
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as WCS
from matplotlib import pyplot as plt

dir = "/Users/rcooke/Work/Research/Accel/data/TW_Hya/2016.1.00440.S/science_goal.uid___A001_X889_X18e/group.uid___A001_X889_X18f/member.uid___A001_X889_X190/product/"
fname = "TW_Hya_NO-tclean.image.pbcor.copy.fits"

dir = "/Users/rcooke/Desktop/datatrans/"
fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.copy.fits"
freq0 = 342.882857*1.0E9

plot = dict(rms=True,
            sig=True,
            velo=True)

# Load cube
dfil = fits.open(dir+fname)
data = dfil[0].data.T[:, :, :, 0]
dsh = data.shape
psh = (dsh[0], dsh[1], 1,)
print("Is it correct to transpose?")

# Formulate the WCS
w = WCS.WCS(dfil[0].header)
coord = np.array([[0, 0, 1, 0]]).repeat(dsh[2], axis=0)
coord[:, 2] = np.arange(coord.shape[0])
world = w.wcs_pix2world(coord, 1)
freq = world[:, 2]
velo = 299792.458*(freq - freq0)/freq0

# Calculate the median and the rms
median = np.median(data, axis=2)
mad = np.median(np.abs(data-median.reshape(psh)), axis=2)
rms = 1.4826*mad
sigmap = data/rms.reshape(psh)

# Plot the median and rms fluctuations
if plot["rms"]:
    plt.subplot(121)
    plt.imshow(median)
    plt.subplot(122)
    plt.imshow(rms)
    plt.show()

# Initialise the intensity, velocity and dispersion maps
imap = np.zeros((dsh[0], dsh[1], 2))
vmap = np.zeros((dsh[0], dsh[1], 2))
wmap = np.zeros((dsh[0], dsh[1], 2))

# Extract data to be used in the moment maps
sigmap[np.isnan(sigmap)] = 0.0  # Remove nans
idx = np.unravel_index(np.argmax(sigmap), dsh)
nspat = 50
nspec = 60
idx_min, idx_max = idx[2] - nspec, idx[2] + nspec
if idx_min <= 0:
    idx_min = 0
if idx_max >= dsh[1]:
    idx_max = dsh[1]
print(idx_min, idx_max)
datacut = data[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_min:idx_max]
velocut = velo[idx_min:idx_max]
norm = np.sum(datacut, axis=2)
datanrm = datacut/norm.reshape(datacut.shape[0], datacut.shape[1], 1)
rmsnrm = rms[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat]/norm

# Calculate the moment maps
dnsh = datanrm.shape
nsim = 100
print(dnsh[0]*dnsh[1]*dnsh[2]*nsim)
xcen = np.zeros((dnsh[0], dnsh[1], dnsh[2], nsim))
xwid = np.ones((dnsh[0], dnsh[1], dnsh[2], nsim))
sims = np.random.normal(xcen, xwid)
sims *= rmsnrm.reshape((dnsh[0], dnsh[1], 1, 1,))
sims += datanrm.reshape((dnsh[0], dnsh[1], dnsh[2], 1,))

mom1 = np.sum(sims * velocut.reshape((1, 1, dnsh[2], 1)), axis=2)
velomap = np.mean(mom1, axis=2)
veloerr = np.std(mom1, axis=2)
if plot["velo"]:
    plt.clf()
    plt.subplot(121)
    plt.imshow(velomap, vmin=-3.5, vmax=-2.2)
    plt.subplot(122)
    plt.imshow(veloerr, vmin=0.1, vmax=0.5)
    plt.show()

# Put the results back onto the original grid
vmap[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, 0] = velomap
vmap[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, 1] = veloerr

# Save the results to a fits file
header = w.to_header()
vmaphdu = fits.PrimaryHDU(vmap, header=header)
vmaphdu.writeto(dir+fname.split(".fits")[0] + '.vmap.fits')
