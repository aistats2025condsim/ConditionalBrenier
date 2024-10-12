#!pip install pykeops > install.log


import os
os.environ['CUDA_PATH'] = '/public/apps/cuda/11.7'

import pykeops
import torch
###necessary because of a "deprecated" issue in the pykeops code that obstructs the output in the experiments below
###if you're ok with seeing these come up, you can comment out the this line
pykeops.set_verbose(False)

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch import generic_sum
    from pykeops.torch import Genred
    keops_available = True
except:
    keops_available = False

#Use gpu if available
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor

print(use_cuda)
print(keops_available)

import sys

import time
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm


import sys
# sys.path.append('/MyDrive/pykeops_test/')

sys.path.insert(1,'../data/')
from datasets import Deterministic_lotka_volterra

sys.path.insert(1,'../')
from visualization import hist_matrix
import scipy.io as sio

sys.path.insert(1,'../methods/')
from entropic_estimator_pykeeops import T, T_epsn, T_D_epsn

import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm

##########################################################
##########################################################
total_sample_size = 100000
data_available = True
target = Deterministic_lotka_volterra(T=20)
dim_x  = target.d
dim_y  = 18

# define samples
if data_available == False:
    model_data = target.sample_joint(N=total_sample_size)
    torch.save(model_data, './samples.pt')
else:
    # load data
    model_data = torch.load('./samples.pt')

# check moments of data
print(torch.mean(model_data))
print(torch.std(model_data))

# discard outliers
non_outlier_target_idx = torch.where(torch.sum(torch.abs(model_data),axis=1) < 1e4)[0]
model_data = model_data[non_outlier_target_idx,:]

# check moments of data
print(torch.mean(model_data))
print(torch.std(model_data))

# print sizes
print(model_data.shape)

# normalize data
data_mean = torch.mean(model_data,axis=0)
data_std  = torch.std(model_data,axis=0)
model_data = (model_data - data_mean[None,:])/data_std[None,:]

# split into x and y
model_x, model_y = model_data[:,:dim_x], model_data[:,dim_x:]
target_data = torch.hstack((model_y, model_x))
source_data = torch.hstack((model_y, torch.randn(model_y.shape[0], dim_x)))


def scale_data(x,beta_vect):
    return x * torch.sqrt(beta_vect)[None,:]

def unscale_map(z,beta_vect):
    return z * torch.sqrt(1./beta_vect)[None,:]

beta = 0.1
eps = 1
beta_vector = torch.ones((dim_x+dim_y))
beta_vector[dim_x:] *= beta
N = len(source_data)

source_data_scaled = scale_data(source_data,beta_vector).type(dtype)
target_data_scaled = scale_data(target_data,beta_vector).type(dtype)

one_over_n = 1/N*torch.ones(source_data.shape[0]).type(dtype)

print(type(source_data_scaled))

u_ab, v_ab = T(eps, one_over_n, source_data_scaled, one_over_n, target_data_scaled, Nits_max = None, keops_available = True)
u_aa, v_aa = T(eps, one_over_n, source_data_scaled, one_over_n, source_data_scaled, Nits_max = None, keops_available = True)

nsamples = 50000
# load data
data = sio.loadmat('problem_data.mat')
yobs = (data['yobs'] - data_mean[dim_x:].numpy())/data_std[dim_x:].numpy()
xtrue = data['xtrue'][0,:]

# generate samples
x1 = torch.tensor(np.tile(yobs.T, nsamples).T)
rho2_given_1 = torch.randn((nsamples, dim_x))
joint_samples = torch.hstack((x1,rho2_given_1)).cpu().double()
gen_cond_samples = (unscale_map(T_epsn(scale_data(joint_samples,beta_vector),
                                       target_data_scaled.cpu().double(),v_ab.cpu().double(),eps).T,beta_vector))
gen_cond_samples_debiased = (unscale_map(T_D_epsn(scale_data(joint_samples,beta_vector),
                                       source_data_scaled.cpu().double(),target_data_scaled.cpu().double(),v_ab.cpu().double(),v_aa.cpu().double(),eps).T,beta_vector))


x_samples = gen_cond_samples[:,dim_y:]
x_samples = x_samples*np.array(data_std[:dim_x]) + np.array(data_mean[:dim_x])

x_samples_debiased = gen_cond_samples_debiased[:,dim_y:]
x_samples_debiased = x_samples_debiased*np.array(data_std[:dim_x]) + np.array(data_mean[:dim_x])
#x_samples = np.array(x_samples.tolist())
# define plotting parameters
symbols = [r'$\alpha$',r'$\beta$',r'$\gamma$',r'$\delta$']
limits = [[0.5,1.2],[0.02,0.07],[0.7,1.5],[0.03,0.07]]

# plot samples
hist_matrix(x_samples, symbols, limits, xtrue,savefig=True,title='biased_eps1.png')

hist_matrix(x_samples_debiased, symbols, limits, xtrue,savefig=True,title='debiased_eps1.png')
