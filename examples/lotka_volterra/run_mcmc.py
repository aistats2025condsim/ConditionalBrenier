import numpy as np
from scipy.optimize import minimize
import scipy.io

import sys
sys.path.append('../')
from examples.datasets import Deterministic_lotka_volterra

from MCMCSamplers import *

if __name__=='__main__':

    LV = Deterministic_lotka_volterra(T=20)
    data = scipy.io.loadmat('problem_data.mat')
    theta = data['xtrue']
    yobs = data['yobs']

    pi = Posterior(yobs[0,:],LV)
    bounds = np.array([[0.]*LV.d,[np.inf]*LV.d])
    x0 = theta
    neg_post = lambda x:-1*pi.logpdf(x)
    xmap = minimize(neg_post,x0)

    prop_std = 0.1
    prop = GaussianProposal(cov=prop_std**2*np.eye(LV.d))

    n_steps = int(1e5)
    mcmc = AdaptiveMetropolisSampler(pi,prop)
    x_samps,_ = mcmc.sample(xmap.x, n_steps, bounds)

    np.savez('mcmc_samps.npz',x_samps=x_samps)

