import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import plotting_routines as pr
from KinMS import *
from kinmspy import make_model

dir = "/Users/rcooke/Work/Research/Cosmo/SandageTest/ALMA/data/TWHya/"
fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.fits"

burnin = 70
ndim = 7
chains = np.load('chains.npy')
samples = chains[:, burnin:, :].reshape((-1, ndim))

# Load the data
cutname = dir+fname.replace(".fits", ".cut.fits")
file = fits.open(cutname)
datcut = file[0].data
velcut = file[1].data
rmscut = file[2].data
rad = file[3].data
sbprof = file[4].data
obspars = dict({})
obspars['xsize'] = file[0].header['xsize']
obspars['ysize'] = file[0].header['ysize']
obspars['vsize'] = file[0].header['vsize']
obspars['cellsize'] = file[0].header['cellsize']
obspars['dv'] = file[0].header['dv']
obspars['beamsize'] = [file[0].header['bmaj'], file[0].header['bmin'], file[0].header['bpa']]
obspars['nsamps'] = file[0].header['nsamps']
obspars['sbprof'] = sbprof
obspars['rms'] = rmscut

# Compute the quantiles.
intflux, posang, inc, centx, centy, voffset, masscen = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                           zip(*np.percentile(samples, [16, 50, 84], axis=0)))

print("""MCMC result:
    par0 = {0[0]} +{0[1]} -{0[2]})
    par1 = {1[0]} +{1[1]} -{1[2]})
    par2 = {2[0]} +{2[1]} -{2[2]})
    par3 = {3[0]} +{3[1]} -{3[2]})
    par4 = {4[0]} +{4[1]} -{4[2]})
    par5 = {5[0]} +{5[1]} -{5[2]})
    par6 = {6[0]} +{6[1]} -{6[2]})
""".format(intflux, posang, inc, centx, centy, voffset, masscen))

param = np.array([intflux[0], posang[0], inc[0], centx[0], centy[0], voffset[0], masscen[0]])

vSize = obspars['vsize']/obspars['dv']
vlos = velcut[0] + (vSize / 2.) + voffset
print("v_LOS=", np.percentile(vlos, [16, 50, 84]))

# Generate the best model
fsim = make_model(param, obspars, rad)
