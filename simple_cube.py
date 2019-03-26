import os
import pdb
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as WCS

dir = "/Users/rcooke/Work/Research/Accel/data/TW_Hya/2016.1.00440.S/science_goal.uid___A001_X889_X18e/group.uid___A001_X889_X18f/member.uid___A001_X889_X190/product/"
fname = "TW_Hya_NO-tclean.image.pbcor.copy.fits"

# Load cube
dfil = fits.open(dir+fname)

w = WCS.WCS(dfil[0].header)

coord = np.array([[100, 100, 5, 0]])
world = w.wcs_pix2world(coord, 1)
print(world)
