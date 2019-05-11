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

# Load some subsampled data
if os.path.exists("datacut_subpix.npy"):
    datsub = np.load("datacut_subpix.npy")
    subarr0 = np.load("datacut_subarr0.npy")
    subarr1 = np.load("datacut_subarr1.npy")
    subarr2 = np.load("datacut_subarr2.npy")


def make_model(fdata, param, obspars):
    # Take a random sample of the input distribution
    sampval = obspars['nsamps'] * datsub
    dsamp = np.floor(sampval)
    dextr = (sampval-dsamp)
    rands = np.random.uniform(0.0, 1.0, dextr.shape)
    ww = np.where(dextr > rands)
    dsamp[ww] += 1
    ww = np.where(dsamp != 0.0)
    dget = dsamp[ww].astype(np.int)
    xvals = subarr0[ww]
    yvals = subarr1[ww]
    vvals = subarr2[ww]
    xout = np.repeat(xvals, dget)
    yout = np.repeat(yvals, dget)
    vout = np.repeat(vvals, dget)
    # xout = np.array([])
    # yout = np.array([])
    # vout = np.array([])
    # for ii in range(dget.size):
    #     xout = np.append(xout, xvals[ii]*np.ones(dget[ii]))
    #     yout = np.append(yout, yvals[ii]*np.ones(dget[ii]))
    #     vout = np.append(vout, vvals[ii]*np.ones(dget[ii]))
    # Perturb data within pixels
    xout += np.random.uniform(-1.0/2.0, 1.0/2.0, xout.size)
    yout += np.random.uniform(-1.0/2.0, 1.0/2.0, yout.size)
    vout += np.random.uniform(-obspars['dv']/2.0, obspars['dv']/2.0, vout.size)

    # Flip the clouds about the axis
    xout = param[0] - xout
    yout = param[1] - yout
    vout = param[2] - vout

    # Rebin these to the old model shape
    mask = np.zeros(fdata.shape)  # =1 when a coordinate value should be included in the fit
    model = np.zeros(fdata.shape)
    # First generate the indices
    xidx = np.int(xout + 0.5)
    yidx = np.int(yout + 0.5)
    vidx = np.int()
    # If any of these values go out of bound, set the mask accordingly
    mask
    # Extract only the indices that are in bounds

    # Finally, bin the result
    np.add.at(model, (xidx, yidx, vidx,), 1)

    # Return the flipped model
    return model, mask


def lnlike(param, obspars, fdata):
    # This function calculates the log-likelihood, comparing model and data

    # Run make_model to produce a model cube
    modout, mskout = make_model(fdata, param, obspars)

    # calculate the chi^2
    chiconv = (mskout*(((fdata - modout) ** 2) / ((obspars['rms']) ** 2))).sum()

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
    param = np.array([centx, centy, voffset])
    #param = np.array([intflux, posang, inc, centx, centy, voffset, masscen, gassigma, rc, gamma])

    # Setup array for priors - in this code all priors are uniform #
    priorarr = np.zeros((param.size, 2))
    priorarr[:, 0] = [mincentx, mincenty, minvoffset]  # Minimum
    priorarr[:, 1] = [maxcentx, maxcenty, maxvoffset]  # Maximum

    # Define the function
    regenerate = False
    if regenerate:
        pts = [np.arange(datacut.shape[0]), np.arange(datacut.shape[1]), obspars['velocut']]
        datfunc = RegularGridInterpolator(pts, datacut, method='linear', bounds_error=False, fill_value=None)  # fill_value=None means extrapolate
        # Subpixellate
        subpix = [3, 3, 1]
        subdsh = (subpix[0]*datacut.shape[0], subpix[1]*datacut.shape[1], subpix[2]*datacut.shape[2],)
        subarr = [None for all in subpix]
        for idx in range(len(subpix)):
            pxsz = pts[idx][1]-pts[idx][0]
            subpts = np.arange(pxsz/(2*subpix[0]), pxsz, pxsz/subpix[0]) - pxsz/2
            subarr[idx] = (pts[idx][:, np.newaxis] + subpts).flatten()
        print("Constructing mesh grid")
        samgrd = np.meshgrid(subarr[0], subarr[1], subarr[2], indexing='ij')
        print("Stacking")
        vst = np.vstack([samgrd[0].flatten(), samgrd[1].flatten(), samgrd[2].flatten()]).T
        print("Evaluating subpixelled model")
        datsub = datfunc(vst).reshape(subdsh)
        print("Normalising and saving")
        # Normalise data
        datsub /= np.sum(datsub)
        np.save("datacut_subpix", datsub)
        np.save("datacut_subarr0", subarr[0])
        np.save("datacut_subarr1", subarr[1])
        np.save("datacut_subarr2", subarr[2])
    return datacut, param, obspars, priorarr


def run_mcmc(datacut, param, obspars, priorarr):
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
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import PPoly
    if delta is None:
        delta = xarr[1]-xarr[0]
    # Initialise the coefficients array
    coeffs = np.zeros((3, xarr.size))
    # Determine the b coeffs
    dd = diags([1, 4, 1], [-1, 0, 1], shape=(xarr.size - 2, xarr.size - 2))
    rvec = 6.0*(yarr[1:-1]-yarr[:-2])/delta**2
    coeffs[1, 1:-1] = spsolve(dd, rvec)
    coeffs[1, 0] = 0.0
    coeffs[1, -1] = 0.0
    # Determine the a coeffs
    coeffs[0, :-1] = 0.5*(coeffs[1, 1:] - coeffs[1, :-1])/delta
    # Determine the c coeffs
    coeffs[2, :-1] = yarr[:-1]/delta - coeffs[0, :-1]*delta**2/3.0 - coeffs[1, :-1]*delta/2.0
    # Get the final coefficnents
    coeffs[2, -1] = coeffs[2, -2] + delta*coeffs[1, -2] + delta**2 * coeffs[0, -2]
    coeffs[0, -1] = (3.0/delta**3) * (yarr[-1] - coeffs[2, -1]*delta - 0.5*coeffs[1, -1]*delta**2)
    # Construct the polynomial
    spl = PPoly(coeffs, np.append(xarr-delta/2, xarr[-1]+delta/2))
    yint = spl(xint)
    pdb.set_trace()
    print(yarr/delta - spl(xarr))
    plt.plot(xarr, yarr/delta, 'bx')
    plt.plot(xint, yint, 'r-')
    plt.show()
    return yint


def test_spline_int():
    from scipy.special import erf
    gmn = 4.678
    gsg = 1.0
    # Make the data
    #xarr = (np.arange(50)/5 + 1)
    xarr = np.arange(10) + 1
    delta = 0.5*(xarr[1]-xarr[0])
    # Generate the true values
    xtru = np.linspace(xarr[0]-delta, xarr[-1]+delta, 10000)
    ytru = np.exp(-0.5*((xtru-gmn)/gsg)**2)
    # Make the integrated spline
    yint = np.zeros(xarr.size)
    for xx in range(xarr.size):
        xu = (xarr[xx]+delta - gmn)/(np.sqrt(2)*gsg)
        xl = (xarr[xx]-delta - gmn)/(np.sqrt(2)*gsg)
        yint[xx] = gsg * np.sqrt(np.pi/2.0) * (erf(xu) - erf(xl))
    yspl = make_spline_int(xtru, xarr, yint)
    # Make the traditional cubic spline
    yarr = np.exp(-0.5 * ((xarr - gmn) / gsg) ** 2)
    spl = UnivariateSpline(xarr, yarr, k=2, s=0, bbox=[xarr[0]-delta, xarr[-1]+delta])
    pdb.set_trace()
    # Plot up a comparison
    plt.plot(xtru, ytru, 'r-')
    plt.plot(xtru, yspl, 'k--')
    plt.plot(xtru, spl(xtru), 'b-')
    plt.plot(xarr, yarr, 'mx')
    print(gmn, xtru[np.argmax(yspl)], xtru[np.argmax(spl(xtru))])
    plt.show()


if __name__ == "__main__":
    #test_spline_int()
    #assert(False)
    mcmc = False
    print("Preparing data...")
    datacut, param, obspars, priorarr = prep_data_model()
    print("complete")
    if mcmc:
        print("Running MCMC")
        run_mcmc(datacut, param, obspars, priorarr)
