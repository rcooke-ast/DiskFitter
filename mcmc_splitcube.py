"""
The idea of this code is to fit one half of the datacube with the other to infer:
(1) the position angle of the disk
(2) the spatial x and y of the disk centre
(3) the velocity of the object along the line of sight
"""

import pdb
import numpy as np
import emcee
from KinMS import *
import os.path
import sys
import time
import multiprocessing
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
import astropy.wcs as WCS
import astropy.io.fits as fits
import astropy.units as u
from scipy import interpolate
from matplotlib import pyplot as plt

Gcons = 6.67408e-11 * u.m**3 / u.kg / u.s**2
dist = 59.5 * u.pc


def make_spline(xarr, yarr, dx=None, convcrit=1.0E-3):
    """
    Make a spline representation of the input arrays, where the spline
    represents the integral from x-dx/2 to x+dx/2
    """
    if dx is None:
        dx = np.ones(xarr.size)*(xarr[1]-xarr[0])
    xsz = xarr.size
    xl = xarr - dx/2.0
    xu = np.roll(xl, -1)
    xu[-1] = xarr[-1] + dx[-1]/2.0
    # Iteratively update the spline
    ynew = yarr.copy()
    yint = yarr.copy()
    iter=0
    while True:
        # Create the spline
        spl = UnivariateSpline(xarr, ynew, k=3, bbox=[xl[0], xu[-1]])
        # Calculate the integral
        for xx in range(xsz):
            yint[xx] = spl.integral(xl[xx], xu[xx])
        # Update the values
        corr = yarr-yint
        ynew += corr*(xu-xl)
        print(iter, np.max(np.abs(corr/yarr)))
        pdb.set_trace()
        if np.all(np.abs(corr/yarr) < convcrit):
            break
        iter += 1
        if iter >= 100:
            pdb.set_trace()
    return UnivariateSpline(xarr, ynew, k=3, bbox=[xl[0], xu[-1]])


def make_model(fdata, param, obspars):
    # Take a random sample of the input distribution
    dsamp = np.round(obspars['nsamps'] * fdata).astype(np.int)
    ww = np.where(dsamp != 0.0)
    dget = dsamp[ww]
    xvals, yvals, vvals = [], [], []
    print("ERROR :: Need to fix subarr line below")
    assert(False)
    subarr = 0.0
    for ii in range(dget.size):
        xvals += dget[ii] * []
        yvals += dget[ii] * []
        vvals += dget[ii] * []
    nval = np.sum(dsamp[ww])
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    vvals = np.array(vvals)

    # This returns the model
    return model


def lnlike(param, obspars, fdata):
    # This function calculates the log-likelihood, comparing model and data

    # Run make_model to produce a model cube
    modout = make_model(fdata, param, obspars)

    # calculate the chi^2
    chiconv = (((fdata - modout) ** 2) / ((obspars['rms']) ** 2)).sum()

    # covert to log-likelihood
    like = -0.5 * (chiconv - fdata.size)
    return like


def priors(param, priorarr):
    # This function checks if any guess is out of range of our priors.

    # initally assume all guesses in range 
    outofrange = 0

    # Loop over each parameter
    for ii in range(0, priorarr[:, 0].size):
        # If the parameter is out of range of the prior then add one to out of range, otherwise add zero
        outofrange += 1 - (priorarr[ii, 0] <= param[ii] <= priorarr[ii, 1])

    if outofrange:
        # If outofrange NE zero at the end of the loop then at least oen parameter is bad, return -inf.
        return -np.inf
    else:
        # Otherwise return zero
        return 0.0


def lnprob(param, obspars, fdata, priorarr):
    # This function calls the others above, first checking that params are valid,
    # and if so returning the log-likelihood.
    checkprior = priors(param, priorarr)
    if not np.isfinite(checkprior):
        return -np.inf
    return lnlike(param, obspars, fdata)


def load_file(year=2011):
    if year == 2011:
        # 12CO(3-2)  --  2011.0.00399.S
        fname = "TW_Hya_2011.0.00399.S_12CO3-2.fits"
        freq0 = 345.79598990*1.0E9
        nspat = 100
        nspec = 70
    elif year == 2016:
        # CSv0  --  2016.1.00440.S
        fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.fits"
        freq0 = 342.882857*1.0E9
        nspat = 40
        nspec = 70
    elif year == 0:
        # Generated cube
        # 12CO(3-2)
        fname = "gencube.fits"
        freq0 = 345.79598990*1.0E9
        nspat = 100
        nspec = 70
    return fname, freq0, nspat, nspec


def prep_data_model():
    # Load in the observational data
    print("Load data -- Is it correct to transpose?")
    dir = "/Users/rcooke/Work/Research/Cosmo/SandageTest/ALMA/data/TWHya/"
    #fname, freq0, nspat, nspec = load_file(2016)
    #fname, freq0, nspat, nspec = load_file(2011)
    (fname, freq0, nspat, nspec), dir = load_file(0), ""

    dfil = fits.open(dir+fname)
#    fdata = dfil[0].data.T[:, :, :, 0]
    fdata = dfil[0].data.T
    dsh = fdata.shape
    psh = (dsh[0], dsh[1], 1,)

    # Calculate the median and the rms
    rmsname = dir+fname.replace(".fits", ".rms.npy")
    if os.path.exists(rmsname):
        print("loading rms...")
        rms = np.load(rmsname)
    else:
        print("calculate rms (may take some time)...")
        median = np.median(fdata, axis=2)
        mad = np.median(np.abs(fdata-median.reshape(psh)), axis=2)
        rms = (1.4826*mad).reshape(psh)
        np.save(rmsname, rms)

    # Determine the velocity width of a channel
    print("get velocity and spatial intervals")
    w = WCS.WCS(dfil[0].header)
    coord = np.array([[0, 0, 1, 0]]).repeat(dsh[2], axis=0)
    coord[:, 2] = np.arange(coord.shape[0])
    world = w.wcs_pix2world(coord, 1)
    freq = world[:, 2]
    velo = 299792.458*(freq - freq0)/freq0

    # Determine the spatial size of a cell
    coord = np.array([[0, 0, 1, 0], [1, 1, 1, 0]])
    world = w.wcs_pix2world(coord, 1)
    cellsize = 3600.0*abs(world[1, 1]-world[0, 1])

    # Extract just the data around the disk
    print("extract data to be used for fitting")
    sigmap = fdata/rms
    sigmap[np.isnan(sigmap)] = 0.0  # Remove nans
    idx = np.unravel_index(np.argmax(sigmap), dsh)
    idx = (dsh[0]//2, dsh[1]//2, idx[2])
    idx_min, idx_max = idx[2] - nspec, idx[2] + nspec
    if idx_min <= 0:
        idx_min = 0
    if idx_max >= dsh[2]:
        idx_max = dsh[2]
    print("Index of maximum flux:", idx)
    print("Spectrum extracted between indices:", idx_min, idx_max)

    datacut = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_min:idx_max]
    velocut = velo[idx_min:idx_max]
    use_median = False
    if use_median:
        rmscut = rms[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat]
    else:
        print("Using line free regions to determine RMS")
        datrms = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_max:]
        rmscut = np.std(datrms, axis=2).reshape((datrms.shape[0], datrms.shape[1], 1))

    check_rms = False
    if check_rms:
        datrms = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_max:]
        val = np.std(datrms, axis=2)
        print(np.mean(val/rmscut), np.std(val/rmscut))
        print("Looks good to me!")
        pdb.set_trace()

    print("TODO :: Could do a better velocity interval")
    vsize = abs(velocut[-1]-velocut[0])
    dvelo = abs(velocut[velocut.size//2] - velocut[velocut.size//2 - 1])

    # Set the spatial sizes
    xsize = datacut.shape[0]*cellsize
    ysize = datacut.shape[1]*cellsize

    # Save some memory
    print("saving memory")
    dfil.close()
    del fdata, rms, sigmap

    flipvelo = True
    if flipvelo:
        velocut = velocut[::-1]
        datacut = datacut[:, :, ::-1]

    # Setup cube parameters #
    print("Set cube parameters")
    obspars = dict({})
    obspars['xsize'] = xsize  # arcseconds
    obspars['ysize'] = ysize  # arcseconds
    obspars['vsize'] = vsize+dvelo  # km/s
    obspars['cellsize'] = cellsize  # arcseconds/pixel
    obspars['dv'] = dvelo  # km/s/channel
    obspars['nsamps'] = 5e5  # Number of cloudlets to use for KinMS models
    obspars['rms'] = rmscut  # RMS of data
    obspars['velocut'] = velocut

    # Write a fits file containing relevant information
    cutname = dir+fname.replace(".fits", ".splitcube.cut.fits")
    hdr = fits.Header()
    for key in obspars.keys():
        if key in ['rms', 'velocut']:
            continue
        hdr[key] = obspars[key]
    dathdu = fits.PrimaryHDU(datacut, header=hdr)
    velhdu = fits.ImageHDU(velocut)
    rmshdu = fits.ImageHDU(rmscut)
    hdul = fits.HDUList([dathdu, velhdu, rmshdu])
    hdul.writeto(cutname, overwrite=True)
    print("File written:\n{0:s}".format(cutname))

    for key in obspars.keys():
        if key in ['rms', 'sbprof']:
            continue
        print(key, obspars[key])

    # Make guesses for the parameters, and set prior ranges
    labels = ['posang', 'centx', 'centy', 'voffset']  # name of each variable, for plot
    posang = 150.  # Best fit posang.
    minposang = 90.  # Min posang.
    maxposang = 180.  # Max posang.
    centx = 0.0  # Best fit x-pos for kinematic centre
    mincentx = -5.0  # min cent x
    maxcentx = 5.0  # max cent x
    centy = 0.0  # Best fit y-pos for kinematic centre
    mincenty = -5.0  # min cent y
    maxcenty = 5.0  # max cent y
    voffset = 0.0  # Best fit velocity centroid
    minvoffset = -vsize/2  # min velocity centroid
    maxvoffset = +vsize/2  # max velocity centroid

    # starting best guess #
    param = np.array([posang, centx, centy, voffset])
    #param = np.array([intflux, posang, inc, centx, centy, voffset, masscen, gassigma, rc, gamma])

    # Setup array for priors - in this code all priors are uniform #
    priorarr = np.zeros((param.size, 2))
    priorarr[:, 0] = [minposang, mincentx, mincenty, minvoffset]  # Minimum
    priorarr[:, 1] = [maxposang, maxcentx, maxcenty, maxvoffset]  # Maximum

    # Define the function
    pdb.set_trace()
    pts = [np.arange(datacut.shape[0]), np.arange(datacut.shape[1]), obspars['velocut']]
    datfunc = RegularGridInterpolator(pts, datacut, method='linear', bounds_error=False, fill_value=None)  # fill_value=None means extrapolate
    # Subpixellate
    subpix = [2, 2, 2]
    subdsh = (subpix[0]*datacut.shape[0], subpix[1]*datacut.shape[1], subpix[2]*datacut.shape[2],)
    subarr = [None for all in subpix]
    for idx in range(len(subpix)):
        pxsz = pts[idx][1]-pts[idx][0]
        subpts = np.arange(pxsz/(2*subpix[0]), pxsz, pxsz/subpix[0]) - pxsz/2
        subarr[idx] = (pts[idx][:, np.newaxis] + subpts).flatten()
    samgrd = np.meshgrid(subarr[0], subarr[1], subarr[2], indexing='ij')
    vst = np.vstack([samgrd[0].flatten(), samgrd[1].flatten(), samgrd[2].flatten()]).T
    datsub = datfunc(vst).reshape(subdsh)
    # Normalise data
    datsub /= np.sum(datsub)
    return datacut, [datsub, subarr], param, obspars, rad, priorarr


def run_mcmc(datacut, datasub, param, obspars, priorarr):
    # Setup MCMC #
    ndim = param.size  # How many parameters to fit
    nwalkers = 200  # Minimum of 2 walkers per free parameter
    mcmc_steps = 20000  # How many sample you want to take of the PDF. 3000 is reasonable for a test, larger the better for actual parameter estimation.
    nsteps = mcmc_steps / nwalkers  # Each walker must take this many steps to give a total of mcmc_steps steps

    # How many CPUs to use. Here I am allowing half of the CPUs. Change this if you want more/less.
    cpus2use = 6

    # Code to run the MCMC
    t0 = time.time()

    pos = [np.random.uniform(priorarr[:, 0], priorarr[:, 1]) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obspars, datacut, priorarr),
                                    threads=cpus2use)  # Setup the sampler

    # Run the samples, while outputing a progress bar to the screen, and writing progress to the file.
    width = 30
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        n = int((width + 1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    sys.stdout.write("\n")
    t1 = time.time()
    print("It took", t1 - t0, "seconds")

    print("Saving samples")
    np.save("splitcube_chains.npy", sampler.chain)


def test_spline():
    xarr = np.arange(10)+1
    #yarr = 2.0*xarr**2*np.exp(-xarr/10.0) + 1/(1+xarr + xarr**2)
    yarr = 2.0 * xarr ** 2 * np.exp(-xarr / 10.0) + 1 / (1 + xarr + xarr ** 2)
    spl = make_spline(xarr, yarr, convcrit=1.0E-6)
    pdb.set_trace()


def make_spline_int(xint, xarr, yarr, delta=None):
    from scipy.sparse import diags
    from scipy.interpolate import PPoly
    if delta is None:
        delta = 0.5*(xarr[1]-xarr[0])
    # Determine the b coeffs
    b = np.zeros(xarr.size)
    dd = diags([1, 4, 1], [-1, 0, 1], shape=(xarr.size - 2, xarr.size - 2))
    rvec = 1.5*(yarr[2:]-yarr[:-2])/delta**2
    b[1:-1] = np.linalg.solve(dd, rvec)

    return yint


def test_spline_int():
    from scipy.special import erf
    gmn = 4.5
    gsg = 1.0
    # Make the data
    xarr = np.arange(10) + 1
    delta = 0.5*(xarr[1]-xarr[0])
    # Generate the true values
    xtru = np.linspace(xarr[0]-delta, xarr[-1]+delta, 1000)
    ytru = np.exp(-0.5*((xtru-gmn)/gsg)**2)
    # Make the integrated spline
    yint = np.zeros(xarr.size)
    for xx in range(xarr.size):
        xu = (xarr[xx]+delta - gmn)/(np.sqrt(2)*gsg)
        xl = (xarr[xx]-delta - gmn)/(np.sqrt(2)*gsg)
        yint[xx] = gsg * np.sqrt(np.pi/2.0) * (erf(xu) - erf(xl))
    yspl = make_spline_int(xtru, xarr, yint, delta=delta)
    # Make the traditional cubic spline
    yarr = np.exp(-0.5 * ((xarr - gmn) / gsg) ** 2)
    spl = UnivariateSpline(xarr, yarr, k=3, bbox=[xarr[0]-delta, xarr[-1]+delta])
    # Plot up a comparison
    plt.plot(xtru, ytru, 'r-')
    plt.plot(xtru, spl(xtru), 'b-')
    plt.show()



if __name__ == "__main__":
    test_spline()
    assert(False)
    mcmc = False
    print("Preparing data...")
    datacut, datasub, param, obspars, rad, priorarr = prep_data_model()
    print("complete")
    if mcmc:
        print("Running MCMC")
        run_mcmc(datacut, param, obspars, priorarr)
