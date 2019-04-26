import pdb
import copy
import mpfit
import numpy as np
import emcee
from KinMS import *
import os.path
import sys
import time
import multiprocessing
import astropy.wcs as WCS
import astropy.io.fits as fits
import astropy.units as u
from scipy import interpolate
from makeplots import makeplots

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


def lnlike(param, obspars, rad, fdata):
    # This function calculates the log-likelihood, comparing model and data

    # Run make_model to produce a model cube
    modout = make_model(param, obspars, rad)

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


def lnprob(param, obspars, rad, fdata, priorarr):
    # This function calls the others above, first checking that params are valid,
    # and if so returning the log-likelihood.
    checkprior = priors(param, priorarr)
    if not np.isfinite(checkprior):
        return -np.inf
    return lnlike(param, obspars, rad, fdata)


def prep_data_model(plotinitial=False):
    # Load in the observational data
    print("Load data -- Is it correct to transpose?")
    #dir = "/Users/rcooke/Work/Research/Accel/data/TW_Hya/2016.1.00440.S/science_goal.uid___A001_X889_X18e/group.uid___A001_X889_X18f/member.uid___A001_X889_X190/product/"
    dir = "/Users/rcooke/Work/Research/Cosmo/SandageTest/ALMA/data/TWHya/"
    fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.fits"
    dfil = fits.open(dir+fname)
    fdata = dfil[0].data.T[:, :, :, 0]
    dsh = fdata.shape
    psh = (dsh[0], dsh[1], 1,)
    freq0 = 342.882857*1.0E9

    # Get the beamsize
    print("get beamsize")
    bmaj = dfil[0].header['BMAJ']*3600.0
    bmin = dfil[0].header['BMIN']*3600.0
    bpa = dfil[0].header['BPA']

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
    nspat = 50
    nspec = 40
    idx_min, idx_max = idx[2] - nspec, idx[2] + nspec
    if idx_min <= 0:
        idx_min = 0
    if idx_max >= dsh[2]:
        idx_max = dsh[2]
    print("Index of maximum flux:", idx)
    print("Spectrum extracted between indices:", idx_min, idx_max)

    datacut = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_min:idx_max]
    velocut = velo[idx_min:idx_max]
    rmscut = rms[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat]

    print("TODO :: Could do a better velocity interval")
    vsize = abs(velocut[-1]-velocut[0])
    dvelo = abs(velocut[velocut.size//2] - velocut[velocut.size//2 - 1])

    # Set the spatial sizes
    xsize = datacut.shape[0]*cellsize
    ysize = datacut.shape[1]*cellsize

    # Calculate the integrated flux
    psf = makebeam(xsize, ysize, [bmaj/cellsize, bmin/cellsize], rot=bpa)
    sumflux = np.sum(datacut)*dvelo/psf.sum()

    # Save some memory
    print("saving memory")
    dfil.close()
    del fdata, rms, sigmap

    # Setup a radius vector [arcseconds]
    radsamp = 10000
    rad = np.linspace(0., xsize/1.414, radsamp)

    # Generate a surface brightness profile
    #radAU, SBprof = np.loadtxt(dir+"TW_Hya_SBprof_Huang2018_COFig4.dat", unpack=True)
    radAU, SBprof = np.loadtxt(dir+"TW_Hya_SBprof_Huang2018_Fig1.dat", unpack=True)
    # Convert AU into arcsec
    angl = (radAU*u.AU/dist).to(u.pc/u.pc).value  # in radians
    angl *= 3600.0 * (180.0/np.pi)
    sbfunc = interpolate.interp1d(angl, SBprof, kind='linear', bounds_error=False, fill_value=0.0)
    sb_profile = sbfunc(rad)/sbfunc(0.0)

    # Setup cube parameters #
    print("Set cube parameters")
    obspars = dict({})
    obspars['xsize'] = xsize  # arcseconds
    obspars['ysize'] = ysize  # arcseconds
    obspars['vsize'] = vsize+dvelo  # km/s
    obspars['cellsize'] = cellsize  # arcseconds/pixel
    obspars['dv'] = dvelo  # km/s/channel
    obspars['beamsize'] = np.array([bmaj, bmin, bpa])  # (arcsec, arcsec, degrees)
    obspars['nsamps'] = 5e5  # Number of cloudlets to use for KinMS models
    obspars['rms'] = rmscut  # RMS of data
    obspars['sbprof'] = sb_profile  # Surface brightness profile

    # Write a fits file containing relevant information
    cutname = dir+fname.replace(".fits", ".cut.fits")
    hdr = fits.Header()
    for key in obspars.keys():
        if key in ['rms', 'sbprof', 'beamsize']:
            continue
        hdr[key] = obspars[key]
    hdr['bmaj'] = bmaj
    hdr['bmin'] = bmin
    hdr['bpa'] = bpa
    dathdu = fits.PrimaryHDU(datacut, header=hdr)
    velhdu = fits.ImageHDU(velocut)
    rmshdu = fits.ImageHDU(rmscut)
    radhdu = fits.ImageHDU(rad)
    sbfhdu = fits.ImageHDU(sb_profile)
    hdul = fits.HDUList([dathdu, velhdu, rmshdu, radhdu, sbfhdu])
    hdul.writeto(cutname, overwrite=True)
    print("File written:\n{0:s}".format(cutname))

    for key in obspars.keys():
        if key in ['rms', 'sbprof']:
            continue
        print(key, obspars[key])

    # Make guesses for the parameters, and set prior ranges
    labels = ["intflux", "posang", "inc", 'centx', 'centy', 'voffset', "masscen"]  # name of each variable, for plot
    intflux = sumflux  # Best fitting total flux
    minintflux = sumflux/5.0  # lower bound total flux
    maxintflux = sumflux  # upper bound total flux
    posang = 150.  # Best fit posang.
    minposang = 90.  # Min posang.
    maxposang = 180.  # Max posang.
    inc = 10.  # degrees
    mininc = 5.0  # Min inc
    maxinc = 15.0  # Max inc
    centx = 0.0  # Best fit x-pos for kinematic centre
    mincentx = -5.0  # min cent x
    maxcentx = 5.0  # max cent x
    centy = 0.0  # Best fit y-pos for kinematic centre
    mincenty = -5.0  # min cent y
    maxcenty = 5.0  # max cent y
    voffset = 0.0  # Best fit velocity centroid
    minvoffset = -vsize/2  # min velocity centroid
    maxvoffset = +vsize/2  # max velocity centroid
    masscen = 0.8  # masscen
    min_masscen = 0.6  # Lower range masscen
    max_masscen = 1.0  # Upper range masscen

    # starting best guess #
    param = np.array([intflux, posang, inc, centx, centy, voffset, masscen])

    # Setup array for priors - in this code all priors are uniform #
    priorarr = np.zeros((param.size, 2))
    priorarr[:, 0] = [minintflux, minposang, mininc, mincentx, mincenty, minvoffset, min_masscen]  # Minimum
    priorarr[:, 1] = [maxintflux, maxposang, maxinc, maxcentx, maxcenty, maxvoffset, max_masscen]  # Maximum

    # Show what the initial model and data look like
    if plotinitial:
        makeplots(fsim, obspars['xsize'], obspars['ysize'], obspars['vsize'], obspars['cellsize'], obspars['dv'],
                  obspars['beamsize'], rms=obspars['rms'], posang=param[1])
        print("[Initial model - close plot to continue]")

    return datacut, param, obspars, rad, priorarr


def run_mcmc(datacut, param, obspars, rad, priorarr):
    # Setup MCMC #
    ndim = param.size  # How many parameters to fit
    nwalkers = 200  # Minimum of 2 walkers per free parameter
    mcmc_steps = 20000  # How many sample you want to take of the PDF. 3000 is reasonable for a test, larger the better for actual parameter estimation.
    nsteps = mcmc_steps / nwalkers  # Each walker must take this many steps to give a total of mcmc_steps steps

    # How many CPUs to use. Here I am allowing half of the CPUs. Change this if you want more/less.
    cpus2use = 6#multiprocessing.cpu_count() // 2

    # Do a test to estimate how long it will take to run the whole code
    print("Estimating the expected execution time")
    t0 = time.time()
    for i in range(0, 10): fsim = make_model(param, obspars, rad)
    t1 = time.time()
    print("One model takes", ((t1 - t0) / 10.), "seconds")
    print("Total runtime expected with", cpus2use, "processors:", (((t1 - t0) / 10.) * mcmc_steps) / (0.6 * cpus2use))

    # Code to run the MCMC
    t0 = time.time()

    tightball = False
    if tightball:
        # walkers start in tight ball around initial guess
        pos = [param + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    else:
        # walkers sample prior space
        pos = [np.random.uniform(priorarr[:, 0], priorarr[:, 1]) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obspars, rad, datacut, priorarr),
                                    threads=cpus2use)  # Setup the sampler

    # Create a new output file, with the next free number #
    num = 0
    chainstart = "KinMS_MCMCrun"
    chainname = chainstart + str(num) + ".dat"
    while os.path.isfile(chainname):
        num += 1
        chainname = chainstart + str(num) + ".dat"
    f = open(chainname, "w")
    f.close()

    # Run the samples, while outputing a progress bar to the screen, and writing progress to the file.
    width = 30
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        position = result[0]
        n = int((width + 1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        f = open(chainname, "a")
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
        f.close()
    sys.stdout.write("\n")
    t1 = time.time()
    print("It took", t1 - t0, "seconds")

    print("Saving samples")
    np.save("chains.npy", sampler.chain)


def myfunct(p, fjac=None, rad=None, fdata=None, err=None, obspars=None):
    # Run make_model to produce a model cube
    model = make_model(p, obspars, rad).flatten()

    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    return [status, (fdata-model)/err]


def run_chisq(datacut, param, obspars, rad, priorarr):
    #######################################
    #          PREPARE THE FIT
    #######################################

    # Set some reasonable starting conditions
    p0 = param

    # Set some constraints you would like to impose
    param_base = {'value': 0., 'fixed': 0, 'limited': [1, 1], 'limits': [0., 0.]}

    # Make a copy of this 'base' for all of our parameters, and set starting parameters
    param_info = []
    for i in range(len(p0)):
        param_info.append(copy.deepcopy(param_base))
        param_info[i]['value'] = p0[i]
        param_info[i]['limits'] = [priorarr[i, 0], priorarr[i, 1]]

    # Now tell the fitting program what we called our variables
    err = obspars['rms'].repeat(datacut.shape[2], axis=2)
    fa = {'rad': rad, 'fdata': datacut.flatten(), 'err': err.flatten(), 'obspars' : obspars}

    #######################################
    #  PERFORM THE FIT AND PRINT RESULTS
    #######################################

    m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=False)
    if (m.status <= 0):
        print('error message = ', m.errmsg)
    print("param: ", m.params)
    print("error: ", m.perror)


if __name__ == "__main__":
    mcmc = False
    chisq = True
    print("Preparing data...")
    datacut, param, obspars, rad, priorarr = prep_data_model()
    print("complete")
    if mcmc:
        print("Running MCMC")
        run_mcmc(datacut, param, obspars, rad, priorarr)
    if chisq:
        print("Running chi-squared minimization")
        run_chisq(datacut, param, obspars, rad, priorarr)
