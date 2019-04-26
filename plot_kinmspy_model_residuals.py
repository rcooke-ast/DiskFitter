import os
import pdb
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
import plotting_routines as pr
from KinMS import *

Gcons = 6.67408e-11 * u.m**3 / u.kg / u.s**2
dist = 59.5 * u.pc


def make_model(param, obspars, rad):
    # rad is in arcseconds

    # Convert input rad [in arcsec] to radians
    rpar = rad * (np.pi/180.0) / 3600.0

    # Use a Keplerian disk, with the central Mass [M_sun] as a free parameter
    Mstar = param[6] * u.Msun / dist
    vel = np.sqrt(Gcons * Mstar / rpar).to(u.km/u.s).value

    # This returns the model
    return KinMS(obspars['xsize'], obspars['ysize'], obspars['vsize'], obspars['cellsize'], obspars['dv'],
                 obspars['beamsize'], param[2], sbProf=obspars['sbprof'], sbRad=rad, velRad=rad, velProf=vel,
                 nSamps=obspars['nsamps'], intFlux=param[0], posAng=param[1], gasSigma=1.,
                 phaseCen=[param[3], param[4]], vOffset=param[5], fixSeed=True)

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
obspars['nsamps'] = 1e7#file[0].header['nsamps']
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

vSize = obspars['vsize']
voffset = np.percentile(samples[:, 5], [16, 50, 84])
print(np.mean(voffset))
vshft = (vSize / 2.) + voffset
vlos = velcut[0] - vshft
vptl = np.percentile(vlos, [16, 50, 84])
vlim = [vptl[1], vptl[2]-vptl[1], vptl[1]-vptl[0]]
print("v_LOS = {0:f} +{1:f} -{2:f}".format(vlim[0], vlim[1], vlim[2]))

# Generate the best model
print("Generating model")
fsim = make_model(param, obspars, rad)
pdb.set_trace()
dathdu = fits.PrimaryHDU(fsim.T)
dathdu.writeto("test.fits", overwrite=True)
dathdu = fits.PrimaryHDU((fsim-datcut).T)
dathdu.writeto("test_diff.fits", overwrite=True)
dathdu = fits.PrimaryHDU(((fsim-datcut)/rmscut).T)
dathdu.writeto("test_resid.fits", overwrite=True)
