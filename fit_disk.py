import sys
import time
import emcee
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as WCS
from scipy.ndimage import rotate as rotate_image


def rebin(arr, shape):
    sh = shape[0], arr.shape[0]//shape[0], shape[1], arr.shape[1]//shape[1]
    return arr.reshape(sh).mean(-1).mean(1)


dir = "/Users/rcooke/Desktop/datatrans/"
fname = "TW_Hya_contsub_CSv0-tclean.image.pbcor.copy.vmap.fits"

# Load cube
dfil = fits.open(dir+fname)
data = dfil[0].data
dsh = data.shape
Gcons = 6.67408e-11 * u.m**3 / u.kg / u.s**2

# Formulate the WCS
w = WCS.WCS(dfil[0].header)
coord = np.array([[0, 0, 1, 0]]).repeat(dsh[2], axis=0)
coord[:, 2] = np.arange(coord.shape[0])
world = w.wcs_pix2world(coord, 1)
dradec = 3600.0*abs(world[1, 1]-world[0, 1])  # Note, this is in arcseconds
#dra = u.arcsec * 3600.0*(world[0, 0]-world[1, 0]) * np.cos(world[0, 1]*np.pi/180.0)

subgrid = 11
y = data[:, :, 0].flatten()
ye = data[:, :, 1].flatten()
x = np.zeros(y.shape)

# Setup the subgrids
xdel, ydel = 0.5, 0.5  # These would be different if using RA and DEC coords
xone = np.linspace(0.0-xdel*(1.0-1.0/subgrid), data.shape[0][-1]+xdel*(1.0-1.0/subgrid), data.shape[0]*subgrid)#.reshape(10,nsubpix).mean(1)
yone = np.linspace(0.0-ydel*(1.0-1.0/subgrid), data.shape[1][-1]+ydel*(1.0-1.0/subgrid), data.shape[1]*subgrid)
xx, yy = np.meshgrid(xone, yone)

# Set the priors
mn_vlsr, mx_vlsr = -3.5, -2.2
mn_Mstar, mx_Mstar = 0.6, 1.0
mn_xcen, mx_xcen = 0.0, data.shape[0]
mn_ycen, mx_ycen = 0.0, data.shape[1]
mn_ang_i, mx_ang_i = 0.0, np.pi/2.0
mn_theta, mx_theta = 0.0, np.pi
mn_dist, mx_dist = 60.0, 60.2

# Some good reference papers:
# https://arxiv.org/pdf/1801.03948.pdf
# https://iopscience.iop.org/article/10.1088/0004-637X/774/1/16/pdf
# https://www.aanda.org/articles/aa/pdf/2006/11/aahk032.pdf
# http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1986MNRAS.218..761H&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
#
# List of model parameters:
# vlsr  = line of sight velocity
# Mstar = central mass
# xcen  = x-position of central mass
# ycen  = y-position of central mass
# ang_i = angle of incidence
# theta = angle on the observer's sky where the disk plane intersects the plane of the sky
# dist  = Distance from observer to the disk
# -------------------------------
# Multi epoch parameters
# pm_x  = proper motion in x
# pm_y  = proper motion in y
# accel = line of sight acceleration --  Delta[vlsr] / Delta[t_obs]

# Coordinates
# disk plane ==> observer plane (see Rosenfeld (2013), ApJ, 774, 16; Eq 2-3)
# x ==> x'
# y ==> y'/cos(i)
# z ==> 0
# theta = arctan(x/y)
# radius = sqrt(x**2 + y**2)
# NOTE: x and x' both point along the major axis of the disk, according to the observer

# Equations
# vobs(x',y') = vlsr + sqrt(G*Mstar/radius) * np.sin(ang_i) * np.cos(theta)
# Note, need to determine where xd points towards at every interation (i.e. cd must point towards the major axis [theta], which is determined at each iteration)
# x', y' ==> xd, yd (where xd and yd are now in arcsecs on the sky)
# vlsr ==> vlsr0 + accel*deltaT
# xcen ==> xcen0 + pm_x
# ycen ==> ycen0 + pm_y
# radius ==> dist * sqrt((xd-xcen)**2 + ((yd-ycen)/cos(ang_i))**2)
# theta ==> arctan((xd-xcen)*cos(ang_i)/(yd-ycen))

def get_model(par):
    vlsr, Mstar, xcen, ycen, ang_i, theta, dist = par
    dist = dist * u.pc
    Mstar = Mstar * u.Msun
    model = vlsr*np.ones((data.shape[0], data.shape[1]))
    xd = rotate_image(xx, theta*180.0/np.pi)
    yd = rotate_image(yy, theta*180.0/np.pi)
    radius = dist * dradec * np.sqrt((xd-xcen)**2 + ((yd-ycen)/np.cos(ang_i))**2)
    model += np.sqrt(Gcons * Mstar / radius) * np.sin(ang_i) * np.cos(theta)
    model = model.to(u.km/u.s).value
    modret = rebin(model, (data.shape[0], data.shape[1]))
    return modret


# Define the probability function as likelihood * prior.
def lnprior(par):
    # vlsr  = line of sight velocity
    # Mstar = central mass
    # xcen  = x-position of central mass
    # ycen  = y-position of central mass
    # ang_i = angle of incidence
    # theta = angle on the observer's sky where the disk plane intersects the plane of the sky
    # dist  = Distance from observer to the disk
    vlsr, Mstar, xcen, ycen, ang_i, theta, dist = par
    if mn_vlsr <= vlsr <= mx_vlsr and \
       mn_Mstar <= Mstar <= mx_Mstar and \
       mn_xcen <= xcen <= mx_xcen and \
       mn_ycen <= ycen <= mx_ycen and \
       mn_ang_i <= ang_i <= mx_ang_i and \
       mn_theta <= theta <= mx_theta and \
       mn_dist <= dist <= mx_dist:
        return 0.0
    return -np.inf


def lnlike(par, x, y, yerr):
    model = get_model(par).flatten()
    inv_sigma2 = 1.0/yerr**2
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Find the maximum likelihood value by brute force.
chi = np.zeros((model_yp.size, Ncol))
for i in range(Ncol):
    chi[:, i] = (y[i]-value_cden[i])/ye[i]
chi2 = chi**2
chisq = np.sum(chi2, axis=1)
bst = np.argsort(chisq)
printbst = 1
for i in range(printbst):
    print("""------------------------\n
        Maximum likelihood result {5:d}/{6:d} {7:.4f}:\n
        [M/H]  = {0}\n
        yp     = {1}\n
        n(H)   = {2}\n
        N(H I) = {3}\n
        slope  = {4}\n""".format(model_met[bst[i]], model_yp[bst[i]], model_hden[bst[i]], model_NHI[bst[i]],
                                 model_slp[bst[i]], i+1, printbst, chisq[bst[i]]))
modvals = [0.0, model_slp[bst[i]], model_met[bst[i]], model_yp[bst[i]], model_hden[bst[i]], model_NHI[bst[i]]]

# Set up the sampler.
ndim, nwalkers = 6, 100
# maxlike = np.array([model_ms[bst], model_ex[bst], model_mx[bst]])
# minv_ms, maxv_ms = 19.0, 22.0
# minv_ex, maxv_ex = 1.0, 5.0
# minv_mx, maxv_mx = 0.0, 0.0001
# minv_ms, maxv_ms = np.min(model_ms[bst[:printbst]]), np.max(model_ms[bst[:printbst]])
# minv_ex, maxv_ex = np.min(model_ex[bst[:printbst]]), np.max(model_ex[bst[:printbst]])
# minv_mx, maxv_mx = np.min(model_mx[bst[:printbst]]), np.max(model_mx[bst[:printbst]])
pos = [np.array([np.random.uniform(mn_sic, mx_sic),
                 np.random.uniform(mn_slp, mx_slp),
                 np.random.uniform(mn_met, mx_met),
                 np.random.uniform(mn_yp, mx_yp),
                 np.random.uniform(mn_hden, mx_hden),
                 np.random.normal(y[0], ye[0])]) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, ye), threads=ndim)

# Clear and run the production chain.
print("Running MCMC...")
nmbr = 15000
a = time.time()
for i, result in enumerate(sampler.run_mcmc(pos, nmbr, rstate0=np.random.get_state())):
    if True:#(i+1) % 100 == 0:
        print("{0:5.1%}".format(float(i) / nmbr))
print("Done.")
print((time.time()-a)/60.0, 'mins')

print("Saving samples")
np.save("diskfit.npy", sampler.chain)
