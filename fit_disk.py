import sys
import time
import emcee
import numpy as np

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

data = np.load("Velocity.npy")

subgrid = 10
y = data[:,:,0].flatten()
ye = data[:,:,1].flatten()
x = BLAH

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

def get_model(theta):
    model = np.zeros(Ncol)
    for ii in range(Ncol):
        model[ii] = model_cden[ii]([[theta[1:]]])
        if 'Si' in yn[ii]:
            model[ii] += theta[0]
    return model


# Define the probability function as likelihood * prior.
def lnprior(theta):
    r, s, m, yy, n, h = theta
    if mn_met <= m <= mx_met and \
       mn_yp <= yy <= mx_yp and \
       mn_hden <= n <= mx_hden and \
       mn_NHI <= h <= mx_NHI and \
       mn_sic <= r <= mx_sic and \
       mn_slp <= s <= mx_slp:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    model = get_model(theta).flatten()
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
