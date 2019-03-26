import os
import pdb
import numpy as np
from alis.alis import alis
from alis.alsave import save_model as get_alis_string

def make_modeldata(freq, flux, flue, p0):
    if not os.path.exists("tempdata"):
        os.mkdir("tempdata")
    sname = "tempdata/spectrum.dat"
    np.savetxt(sname, np.transpose((freq, flux, flue)))
    parlines = get_parlines()
    datlines = get_datlines(sname)
    modlines = get_modlines(p0)
    return parlines, datlines, modlines

def get_datlines(sname, kind="hanning"):
    datlines = []
    shtxt = "0.0SFIX"
    datlines += ["  {0:s}  specid=0  fitrange=columns  loadrange=all  resolution=apod(kind:{1:s}) shift=vshift({2:s}) columns=[wave:0,flux:1,error:2]".format(sname, res, shtxt)]
    return datlines

def get_modlines(p0):
    modlines = []
    # Do the emission
    modlines += ["emission"]
    modlines += ["legendre  0.0 0.0  scale=1.0,1.0  specid=0"]
    modlines += ["gaussian  {0:f}   {1:f}   {2:f}  specid=0".format(p0[0], p0[1], p0[2])]
    return modlines

def get_parlines():
    parlines = []
    parlines += ["run  ncpus  3"]
    parlines += ["run ngpus 0"]
    parlines += ["run nsubpix 5"]
    parlines += ["run blind False"]
    parlines += ["run convergence False"]
    parlines += ["run convcriteria 0.2"]
    parlines += ["run bintype Hz"]
    parlines += ["chisq atol 0.001"]
    parlines += ["chisq xtol 0.0"]
    parlines += ["chisq ftol 0.0"]
    parlines += ["chisq gtol 0.0"]
    parlines += ["chisq fstep 1.3"]
    parlines += ["chisq miniter 10"]
    parlines += ["chisq maxiter 1000"]
    parlines += ["out model False"]
    parlines += ["out fits False"]
    parlines += ["out verbose -1"]
    parlines += ["out overwrite True"]
    parlines += ["plot fits False"]
    return parlines

def parse_fit(result):
    pars, errs = np.zeros(3), np.zeros(3)
    # Convert the results into an easy read format
    fres = result._fitresults
    info = [(result._tend - result._tstart)/3600.0, fres.fnorm, fres.dof, fres.niter, fres.status]
    alis_lines = get_alis_string(result, fres.params, fres.perror, info,
                                 printout=False, getlines=True, save=False)

    # Scan through the model to get parameter values and errors
    alspl = alis_lines.split("\n")
    for spl in range(len(alspl)):
        if "gaussian" in alspl[spl]:
            # Values
            flag = 0
            if "#" in alspl[spl]:
                flag = 1
        else:
            continue
        # Go through all the lines
        if flag == 0:
            pars[0], pars[1], pars[2] = float(alspl[1]), float(alspl[2]), float(alspl[3])
        elif flag == 1:
            errs[0], errs[1], errs[2] = float(alspl[2]), float(alspl[3]), float(alspl[4])
    return pars, errs

dir = "/Users/rcooke/Work/Research/Accel/data/TW_Hya/2016.1.00440.S/science_goal.uid___A001_X889_X18e/group.uid___A001_X889_X18f/member.uid___A001_X889_X190/product/"
fname = "TW_Hya_NO-tclean.image.pbcor.copy.fits"
filename = dir+fname
xst, yst = 100, 100   # Index of where to start fitting the data
ist, zst, wst = 1000.0, 2.84/299792.458, 1.0E-3   # Initial guesses for starting point

# Load cube
dfil = fits.open(filename)
flxdat = dfil[0].data
fledat = dfil[1].data
freq = 
xsh, ysh = flxdat.shape[0], flxdat.shape[1]

# Create a map of the order in which to fit the results
xx, yy = np.meshgrid(np.arange(xsh)-xst, np.arange(ysh)-yst)
ordmap = np.ma.MaskedArray(np.sqrt(xx**2 + yy**2), mask=np.zeros((xsh, ysh), dtype=np.int))

# Initialise the intensity, velocity and dispersion maps
imap = np.zeros((xsh, ysh, 2))
vmap = np.zeros((xsh, ysh, 2))
wmap = np.zeros((xsh, ysh, 2))

# Perform the first fit
flux = flxdat[xst,yst,:].flatten()
flue = fledat[xst,yst,:].flatten()
p0 = [ist,zst,wst]
parlines, datlines, modlines = make_modeldata(freq, flux, flue, p0)
result = alis(parlines=parlines, datlines=datlines, modlines=modlines, verbose=-1)
pars, errs = parse_fit(result)
imap[xst, yst, 0], vmap[xst, yst, 0], wmap[xst, yst, 0] = pars[0], pars[1], pars[2]
imap[xst, yst, 1], vmap[xst, yst, 1], wmap[xst, yst, 1] = errs[0], errs[1], errs[2]
ordmap[xst, yst] = np.ma.masked

cntr = 1
while cntr != xsh*ysh:
    if cntr%100 == 0:
        print(cntr, xsh*ysh)
    # Find the next indice to fit
    xind, yind = np.unravel_index(np.argmin(ordmap), (xsh,ysh))
    # Extract the data
    flux = flxdat[xind, yind, :].flatten()
    flue = fledat[xind, yind, :].flatten()
    # Find the starting parameters
    xx, yy = np.meshgrid(np.arange(xsh)-xind, np.arange(ysh)-yind)
    stpmap = np.ma.MaskedArray(np.sqrt(xx**2 + yy**2), mask=~ordmap.mask)
    xp, yp = np.unravel_index(np.argmin(ordmap), (xsh,ysh))
    p0 = [imap[xp,yp], vmap[xp,yp], wmap[xp,yp]]
    parlines, datlines, modlines = make_modeldata(freq, flux, flue, p0)
    # Perform the fit
    result = alis(parlines=parlines, datlines=datlines, modlines=modlines, verbose=-1)
    pars, errs = parse_fit(result)
    # Store the result
    imap[xind, yind, 0], vmap[xind, yind, 0], wmap[xind, yind, 0] = pars[0], pars[1], pars[2]
    imap[xind, yind, 1], vmap[xind, yind, 1], wmap[xind, yind, 1] = errs[0], errs[1], errs[2]
    ordmap[xind, yind] = np.ma.masked
    if True:
        # Do some checks along the way to make sure the image is getting filled in correctly
        plt.clf()
        plt.imshow(imap[:,:,0])
        plt.show()
    cntr += 1

# Save the data
np.save("Intensity", imap)
np.save("Velocity", vmap)
np.save("Dispersion", wmap)

# Plot the results
plt.subplot(131)
plt.imshow(imap[:,:,0])
plt.subplot(132)
plt.imshow(vmap[:,:,0])
plt.subplot(133)
plt.imshow(wmap[:,:,0])
plt.show()