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
from matplotlib import pyplot as plt

Gcons = 6.67408e-11 * u.m**3 / u.kg / u.s**2
dist = 59.5 * u.pc


def make_model(param, obspars, rad, ring):
    """rad is in arcseconds
    ring[0] = radius at maximum
    ring[1] = FWHM of ring
    """

    intflux, posang, inc, centx, centy, voffset, masscen, gassigma =\
        param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7]

    # Get the surface brightness profile
    sbProf = 1 + param[8]*(rad-ring[0])
    sbProf[sbProf < 0.0] = 0.0

    # Convert input rad [in arcsec] to radians
    rpar = rad * (np.pi/180.0) / 3600.0

    # Use a Keplerian disk, with the central Mass [M_sun] as a free parameter
    Mstar = masscen * u.Msun / dist
    vel = np.sqrt(Gcons * Mstar / rpar).to(u.km/u.s).value

    # Create weighting
    cellSize = obspars['cellsize']
    xs, ys = obspars['xsize'], obspars['ysize']
    xSize = float(round(xs/cellSize))
    ySize = float(round(ys/cellSize))
    cent = [(xSize / 2.) + (centx / cellSize), (ySize / 2.) + (centy / cellSize)]
    cent = [xs/2, ys/2]
    xx, yy = np.meshgrid(np.linspace(0.0, xs-cellSize, xSize)-cent[0], np.linspace(0.0, ys-cellSize, ySize)-cent[1], indexing='ij')
    radcen = np.sqrt(xx**2 + yy**2)
    #wght = np.cos(0.5 * np.pi * (radcen - ring[0]) / ring[1]) ** 2
    wght = 1 - np.abs(radcen - ring[0])/ring[1]
    wght[(ring[0]-ring[1] > radcen) | (radcen > ring[0]+ring[1])] = 0.0
    # plt.imshow(wght)
    # plt.show()
    # This returns the model
    return wght, KinMS(obspars['xsize'], obspars['ysize'], obspars['vsize'], obspars['cellsize'], obspars['dv'],
                 obspars['beamsize'], inc, sbProf=sbProf, sbRad=rad, velRad=rad, velProf=vel,
                 nSamps=obspars['nsamps'], intFlux=intflux, posAng=posang, gasSigma=gassigma,
                 phaseCen=[centx, centy], vOffset=voffset, fixSeed=True)


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


def load_file(year=2011):
    if year == 2011:
        # 12CO(3-2)  --  2011.0.00399.S
        fname = "TW_Hya_2011.0.00399.S_12CO3-2.fits"
        freq0 = 345.79598990*1.0E9
        # nspat = 100
        # nspec = 70
        nspat = 75
        nspec = 40
        idx = (216, 197, 99)
    elif year == 2016:
        # CSv0  --  2016.1.00440.S
        fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.fits"
        freq0 = 342.882857*1.0E9
        nspat = 40
        nspec = 70
        idx = None
    return fname, freq0, nspat, nspec, idx


def prep_data_model(plotinitial=False, gencube=False):
    # Load in the observational data
    print("Load data -- Is it correct to transpose?")
    dir = "/Users/rcooke/Work/Research/Cosmo/SandageTest/ALMA/data/TWHya/"
    #fname, freq0, nspat, nspec, idx = load_file(2016)
    fname, freq0, nspat, nspec, idx = load_file(2011)

    dfil = fits.open(dir+fname)
    fdata = dfil[0].data.T[:, :, :, 0]
    dsh = fdata.shape
    psh = (dsh[0], dsh[1], 1,)

    #dathdu = fits.PrimaryHDU(fdata.sum(2))
    #dathdu.writeto("intflux.fits", overwrite=True)
    #pdb.set_trace()

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
    if idx is None:
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
        datrms_a = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, :30]
        datrms_b = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, -30:]
        datrms = np.append(datrms_a, datrms_b, axis=2)
        rmscut = np.std(datrms, axis=2).reshape((datrms.shape[0], datrms.shape[1], 1))

    check_rms = False
    if check_rms:
        datrms = fdata[idx[0]-nspat:idx[0]+nspat, idx[1]-nspat:idx[1]+nspat, idx_max:]
        val = np.std(datrms, axis=2)
        print(np.mean(val/rmscut), np.std(val/rmscut))
        print("Looks good to me!")
        pdb.set_trace()

    # Save some memory
    print("saving memory")
    dfil.close()
    del fdata, rms, sigmap

    print("TODO :: Could do a better velocity interval")
    vsize = abs(velocut[-1]-velocut[0])
    dvelo = abs(velocut[velocut.size//2] - velocut[velocut.size//2 - 1])

    # Set the spatial sizes
    xsize = datacut.shape[0]*cellsize
    ysize = datacut.shape[1]*cellsize

    # Calculate the integrated flux
    psf = makebeam(xsize, ysize, [bmaj/cellsize, bmin/cellsize], rot=bpa)
    sumflux = np.sum(datacut)*dvelo/psf.sum()

    # Setup a radius vector [arcseconds]
    radsamp = 10000
    rad = np.linspace(0., xsize/1.414, radsamp)

    # Generate a surface brightness profile
    radAU, SBprof = np.loadtxt(dir+"TW_Hya_SBprof_Huang2018_COFig4.dat", unpack=True)
    #radAU, SBprof = np.loadtxt(dir+"TW_Hya_SBprof_Huang2018_Fig1.dat", unpack=True)
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
    obspars['nsamps'] = 5e6  # Number of cloudlets to use for KinMS models
    obspars['rms'] = rmscut  # RMS of data
    obspars['sbprof'] = sb_profile  # Surface brightness profile
    obspars['velocut0'] = velocut[0]

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
    labels = ["intflux", 'posang', 'inc', 'centx', 'centy', 'voffset', "masscen", "gassigma"]  # name of each variable, for plot
    intflux = sumflux  # Best fitting total flux
    minintflux = 0.0  # lower bound total flux
    maxintflux = 10*sumflux  # upper bound total flux
    posang = 150.  # Best fit posang.
    minposang = 90.  # Min posang.
    maxposang = 230.  # Max posang.
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
    gassigma = 0.2  # gas velocity dispersion
    min_gassigma = 0.0  # Lower range velocity dispersion
    max_gassigma = 2.0  # Upper range velocity dispersion

    rc = 1.0  # arcsec
    min_rc = 0.2  # Lower range masscen
    max_rc = 2.0  # Upper range masscen
    gamma = 3.0  # masscen
    min_gamma = 2.1  # Lower range masscen
    max_gamma = 5.0  # Upper range masscen

    # starting best guess #
    param = np.array([intflux, posang, inc, centx, centy, voffset, masscen, gassigma])
    #param = np.array([intflux, posang, inc, centx, centy, voffset, masscen, gassigma, rc, gamma])

    # Setup array for priors - in this code all priors are uniform #
    priorarr = np.zeros((param.size, 2))
    priorarr[:, 0] = [minintflux, minposang, mininc, mincentx, mincenty, minvoffset, min_masscen, min_gassigma]  # Minimum
    priorarr[:, 1] = [maxintflux, maxposang, maxinc, maxcentx, maxcenty, maxvoffset, max_masscen, max_gassigma]  # Maximum
    # priorarr[:, 0] = [minintflux, minposang, mininc, mincentx, mincenty, minvoffset, min_masscen, min_gassigma, min_rc, min_gamma]  # Minimum
    # priorarr[:, 1] = [maxintflux, maxposang, maxinc, maxcentx, maxcenty, maxvoffset, max_masscen, max_gassigma, max_rc, max_gamma]  # Maximum

    # if gencube:
    #     fsim = make_model(param, obspars, rad)
    #     dathdu = fits.PrimaryHDU(fsim.T, header=dfil[0].header)
    #     dathdu.writeto("gencube.fits", overwrite=True)

    # Show what the initial model and data look like
    if plotinitial:
        param = np.array([38.47846005, 0.0, 0.0, 0.0001134311081, 0.996607506, 0.2810434911])
        fsim = make_model(param, obspars, rad)
        dathdu = fits.PrimaryHDU(fsim.T)
        dathdu.writeto("init_sim.fits", overwrite=True)
        #makeplots(fsim, obspars['xsize'], obspars['ysize'], obspars['vsize'], obspars['cellsize'], obspars['dv'],
        #          obspars['beamsize'], rms=obspars['rms'], posang=param[1])
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


def myfunct_ring(p, fjac=None, rad=None, fdata=None, err=None, obspars=None, ddpid=None, ring=None):
    # Run make_model to produce a model cube
    wght, model = make_model(p, obspars, rad, ring)
    wght = wght[:, :, np.newaxis].repeat(model.shape[2], axis=2).flatten()

    chisq_model = model.flatten()*wght + fdata*(1-wght)

    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    return [status, (fdata-chisq_model)/err]


def run_chisq(datacut, param, obspars, rad, priorarr, rings=None):
    if rings is None:
        rings = np.linspace(0.0, 3.0, 20)

    # Start with a better guess of the disk parameters
    param = np.array([22.44355614, 151.5557589, 5.630761085, 0, -0.02, 0.01462441849, 0.8, 0.1441155137])
    #param = np.array([250.798918, 150.0, 7.214971093, 0.0, -0.02, 0.02, 0.8, 0.4])
    #param = np.array([1.145, 151.189245, 6.969, 0.035626, 0.0344, 0.1513, 0.8, 0.0764])
    steps = np.array([1.0E-3, 1.0E-1, 1.0E-1, 1.0E-4, 1.0E-4, 1.0E-4, 1.0e-3, 1.0E-3, 0.01])
    #p0 = np.append(param, 0.8)  # Add another parameter which represents the gradient of the surface brightness profile
    p0 = np.append(param, 0.0)  # Add another parameter which represents the gradient of the surface brightness profile

    # Set some constraints you would like to impose
    param_base = {'value': 0., 'fixed': 0, 'limited': [1, 1], 'limits': [0., 0.], 'step': 0.0, 'relstep': 0.01}

    # Make a copy of this 'base' for all of our parameters, and set starting parameters
    param_info = []
    for i in range(len(p0)):
        param_info.append(copy.deepcopy(param_base))
        param_info[i]['value'] = p0[i]
        if i < len(param):
            param_info[i]['limits'] = [priorarr[i, 0], priorarr[i, 1]]
        else:
            param_info[i]['limits'] = [-100, 100]
        param_info[i]['step'] = steps[i]
    param_info[3]['fixed'] = 1
    param_info[4]['fixed'] = 1
    param_info[6]['fixed'] = 1
    param_info[8]['fixed'] = 1

    # Set the errors
    #err = obspars['rms'].repeat(datacut.shape[2], axis=2).flatten()
    err = np.mean(obspars['rms'])#.flatten()

    outvals = np.zeros((p0.size, rings.size-1))
    outerrs = np.zeros((p0.size, rings.size-1))
    # gen_wght = True
    # allwght = np.zeros((datacut.shape[0], datacut.shape[1]))
    for rrt in range(rings.size-1):
        rr = rings.size - 2 - rrt
        print(rr+1, rings.size-1)
        # Restart from the previous best parameters
        if rr != rings.size - 2:
            for i in range(len(p0)):
                param_info[i]['value'] = m.params[i]
                p0[i] = m.params[i]
        # Set the ring properties
        ring = [rings[rr], rings[rr+1]-rings[rr]]

        # if gen_wght:
        #     cellSize = obspars['cellsize']
        #     xs, ys = obspars['xsize'], obspars['ysize']
        #     xSize = float(round(xs / cellSize))
        #     ySize = float(round(ys / cellSize))
        #     cent = [xs / 2, ys / 2]
        #     xx, yy = np.meshgrid(np.linspace(0.0, xs - cellSize, xSize) - cent[0],
        #                          np.linspace(0.0, ys - cellSize, ySize) - cent[1], indexing='ij')
        #     radcen = np.sqrt(xx ** 2 + yy ** 2)
        #     # wght = np.cos(0.5 * np.pi * (radcen - ring[0]) / ring[1]) ** 2
        #     wght = 1 - np.abs(radcen - ring[0]) / ring[1]
        #     wght[(ring[0] - ring[1] > radcen) | (radcen > ring[0] + ring[1])] = 0.0
        #     allwght += wght
        #     plt.imshow(wght)
        #     plt.show()
        #     continue

        # Now tell the fitting program what we called our variables
        fa = {'rad': rad, 'fdata': datacut.flatten(), 'err': err, 'obspars': obspars, 'ring': ring}

        #######################################
        #  PERFORM THE FIT AND PRINT RESULTS
        #######################################

        m = mpfit.mpfit(myfunct_ring, p0, parinfo=param_info, functkw=fa, quiet=False, ncpus=5)
        if m.status <= 0:
            print('error message = ', m.errmsg)
        outvals[:, rr] = m.params.copy()
        outerrs[:, rr] = m.perror.copy()
    # plt.imshow(allwght)
    # plt.show()
    return outvals, outerrs


def run_chisq_spi_fixed(datacut, param, obspars, rad, priorarr):
    # Start with a better guess of the disk parameters

    param = np.array([1.145, 0.035626, 0.0344, 0.1513, 0.8, 0.0764])
    steps = np.array([1.0E-4, 1.0E-5, 1.0E-5, 1.0E-5, 1.0e-4, 1.0E-4])*parscale
    p0 = param * parscale

    spldict = dict(sb=True, inc=True, posang=True, gassig=True)
    splmsk = np.zeros(param.size)

    #######################################
    #          PREPARE THE FIT
    #######################################
    if spldict['sb']:
        # Include the radial surface brightness profile as a free parameter
        nsurfb = 10
        sbrad = np.array(
            [0.0, 0.15789474, 0.31578947, 0.47368421, 0.63157895, 0.78947368, 0.94736842, 1.10526316, 1.26315789,
             1.57894737, 1.89473684, 2.21052632, 2.52631579, 2.84210526])
        sb_p0 = np.array(
            [454.12234286, 393.36149741, 286.80762835, 216.90814331, 207.93973294, 161.50987764, 140.72118948,
             117.95176048, 98.90525955, 68.81306633, 48.03526046, 32.32684097, 21.07468516, 14.83180197])
        sb_p0 *= 1.0/np.max(sb_p0)
        # Fill the mask and add to initial parameters
        splmsk = np.append(splmsk, 1*np.ones(nsurfb))
        p0 = np.append(p0, sb_p0)
        priorarr = np.append(priorarr, np.repeat([[0.0, 1.0]], nsurfb, axis=0), axis=0)
        steps = np.append(steps, np.zeros(nsurfb))
    else:
        sbrad = None

    if spldict['posang']:
        # Include the radial position angle profile as a free parameter
        sprad = np.array([0.0, 0.15, 0.3, 0.5, 0.8, 1.15, 1.6, 2.0, 3.0])
        sp_p0 = np.array([172, 160.0, 154.8, 155.4, 152.7, 153.3, 152.5, 152.2, 151.5])
        nspos = sprad.size
        # Fill the mask and add to initial parameters
        splmsk = np.append(splmsk, 2*np.ones(nspos))
        p0 = np.append(p0, sp_p0)
        priorarr = np.append(priorarr, np.repeat([[90.0, 180.0]], nspos, axis=0), axis=0)
        steps = np.append(steps, 0.1*np.ones(nspos))
    else:
        p0 = np.append(p0, 150.0)
        splmsk = np.append(splmsk, 2)
        priorarr = np.append(priorarr, np.repeat([[90.0, 180.0]], 1, axis=0), axis=0)
        steps = np.append(steps, 0.1)
        sprad = None

    if spldict['inc']:
        # Include the inclination profile as a free parameter
        sirad = np.array([0.0, 0.25, 0.50, 1.0, 2.0, 3.0])
        si_p0 = np.array([7.35403098, 7.63412163, 7.70179149, 7.35666027, 5.84205372, 5.11887352])
        nsinc = sirad.size
        # Fill the mask and add to initial parameters
        splmsk = np.append(splmsk, 3*np.ones(nsinc))
        p0 = np.append(p0, si_p0)
        priorarr = np.append(priorarr, np.repeat([[5.0, 15.0]], nsinc, axis=0), axis=0)
        steps = np.append(steps, 0.01*np.ones(nsinc))
    else:
        splmsk = np.append(splmsk, 3)
        p0 = np.append(p0, 10.0)
        priorarr = np.append(priorarr, np.repeat([[5.0, 15.0]], 1, axis=0), axis=0)
        steps = np.append(steps, 0.01)
        sirad = None

    if spldict['gassig']:
        # Include the inclination profile as a free parameter
        sgrad = np.array([0.0, 0.15, 0.3, 0.5, 0.650, 1.0, 2.0, 3.0])
        sg_p0 = np.array([0.078, 0.078, 0.085, 0.13, 0.223, 0.22, 0.17, 0.14])
        nssig = sgrad.size
        # Fill the mask and add to initial parameters
        splmsk = np.append(splmsk, 4*np.ones(nssig))
        p0 = np.append(p0, sg_p0)
        priorarr = np.append(priorarr, np.repeat([[0.5, 0.3]], nssig, axis=0), axis=0)
        steps = np.append(steps, 0.01*np.ones(nssig))
    else:
        splmsk = np.append(splmsk, 4)
        p0 = np.append(p0, 0.2)
        priorarr = np.append(priorarr, np.repeat([[0.5, 0.3]], 1, axis=0), axis=0)
        steps = np.append(steps, 0.001)
        sgrad = None

    # Set some constraints you would like to impose
    param_base = {'value': 0., 'fixed': 0, 'limited': [1, 1], 'limits': [0., 0.], 'step': 0.0, 'relstep': 0.001}

    # Make a copy of this 'base' for all of our parameters, and set starting parameters
    param_info = []
    for i in range(len(p0)):
        param_info.append(copy.deepcopy(param_base))
        param_info[i]['value'] = p0[i]
        if i < len(param):
            param_info[i]['limits'] = [priorarr[i, 0]*parscale[i], priorarr[i, 1]*parscale[i]]
            param_info[i]['step'] = steps[i]
        else:
            param_info[i]['limits'] = [priorarr[i, 0], priorarr[i, 1]]
            param_info[i]['step'] = steps[i]

    # Now tell the fitting program what we called our variables
    #err = obspars['rms'].repeat(datacut.shape[2], axis=2).flatten()
    err = np.mean(obspars['rms'])#.flatten()
    fa = {'spi_rad': [sbrad, sprad, sirad, sgrad], 'spl_msk':splmsk, 'rad': rad, 'fdata': datacut.flatten(), 'err': err, 'obspars': obspars}

    # Do a quick check to make sure the model is being interpreted correctly
    plotinitial = True
    if plotinitial:
        model = make_model(p0, obspars, rad, spi_rad=[sbrad, sprad, sirad, sgrad], spl_msk=splmsk, plotit=True)
        dathdu = fits.PrimaryHDU(model.T)
        dathdu.writeto("test.fits", overwrite=True)

    #######################################
    #  PERFORM THE FIT AND PRINT RESULTS
    #######################################

    m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=False, ncpus=1)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    print("param: ", m.params)
    print("error: ", m.perror)
    fsim = make_model(m.params, obspars, rad, spi_rad=[sbrad, sprad, sirad], spl_msk=splmsk, plotit=True)
    dathdu = fits.PrimaryHDU(fsim.T)
    dathdu.writeto("test.fits", overwrite=True)
    dathdu = fits.PrimaryHDU(datacut.T)
    dathdu.writeto("datacut.fits", overwrite=True)
    dathdu = fits.PrimaryHDU((fsim - datacut).T)
    dathdu.writeto("test_diff.fits", overwrite=True)
    dathdu = fits.PrimaryHDU(((fsim - datacut) / err).T)
    dathdu.writeto("test_resid.fits", overwrite=True)
    if False:
        plt.subplot(211)
        plt.plot(rad, sbProf, 'k-')
        # convert rad to radAU
        plt.subplot(212)
        radAU = (dist * rad / (3600.0 * (180.0/np.pi))).to(u.AU)
        plt.plot(radAU.value, sbProf, 'k-')
        plt.show()
    pdb.set_trace()
    vSize = obspars['vsize']
    vshft = (vSize / 2.) + m.params[3]
    vlos = obspars['velocut0'] - vshft
    return


if __name__ == "__main__":
    mcmc = False
    chisq = True
    gencube = False
    print("Preparing data...")
    datacut, param, obspars, rad, priorarr = prep_data_model(gencube=gencube)
    print("complete")
    if mcmc:
        print("Running MCMC")
        run_mcmc(datacut, param, obspars, rad, priorarr)
    if chisq:
        print("Running chi-squared minimization")
        fitit = True
        if fitit:
            outvals, outerrs = run_chisq(datacut, param, obspars, rad, priorarr)
            np.save("outvals_fixpar_triangle", outvals)
            np.save("outerrs_fixpar_triangle", outerrs)
        else:
            outvals = np.load("outvals_fixpar_triangle.npy")
            outerrs = np.load("outerrs_fixpar_triangle.npy")
        rings = np.linspace(0.0, 3.0, 20)
        for ii in range(outvals.shape[0]):
            plt.subplot(3, 3, ii+1)
            plt.plot(rings[:-1], outvals[ii, :], marker='o', color='b')
            plt.errorbar(rings[:-1], outvals[ii, :], outerrs[ii, :], fmt='none', color='b')
            #plt.errorbar(rings[:-1], outvals[ii, :], outerrs[ii, :], fmt='')
        plt.show()
