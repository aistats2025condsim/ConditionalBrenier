import pykeops

###necessary because of a "deprecated" issue in the pykeops code that obstructs the output in the experiments below
###if you're ok with seeing these come up, you can comment out the this line
pykeops.set_verbose(False)

###necessary imports
#import numpy as np
import torch
from math import *

import matplotlib.pyplot as plt
import statsmodels.api as sm

import pickle
import time

from functools import partial

#####
# Code here is taken from: https://gitlab.com/proussillon/wasserstein-estimation-sinkhorn-divergence with some small modifications
##### 
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
print(dtype)
print(dtypeint)

def log_weights(α):
    α_log = α.log()
    α_log[α <= 0] = -100000
    return α_log

def sinkhorn_loop( softmin, marg_x, α_log, β_log, C_xy, C_yx, ε, Nits_max = None):
#sinkhorn_loop performs the Sinkhorn iterations until convergence.
    Nits = 1
    N, D = C_xy[0].shape
    M, D = C_xy[1].shape
    # Start with a decent initialization for the dual vectors:
    v = softmin(ε, C_yx, α_log )  # OT(α,β) wrt. a
    u = softmin(ε, C_xy, β_log )  # OT(α,β) wrt. b
    
    err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
    err_marginal = [err]

    if Nits_max == None:
        Nits_max = 5*10**3

    while (Nits< Nits_max) and (err > 5*10**(-4) or Nits < 2 ):
        Nits += 1
        # "Coordinate ascent" on the dual problems:
        v = softmin(ε, C_yx, α_log + u/ε )  # OT(α,β) wrt. a
        u = softmin(ε, C_xy, β_log + v/ε )  # OT(α,β) wrt. b
        if Nits % 50 == 0:
          err = (torch.abs(marg_x(ε, C_yx, v, u)/N-1.)).sum()/M
          err_marginal.append(err)
    u_t, v_t = u, v
    return u_t, v_t

def T(eps, alpha, x, beta, y, Nits_max=None,keops_available = keops_available):
    n, d = x.shape
    Loss =  loss_sinkhorn_online(d, p=2, eps=eps,Nits_max = Nits_max, keops_available = keops_available)
    u_ab, v_ab = Loss( alpha, x, beta, y )
            
    return u_ab, v_ab

# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def squared_distances(x, y):
    D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
    D_xy = 2*torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
    D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    return D_xx - D_xy + D_yy

def softmin_tensorized():
    def softmin(ε, C_xy, g):
        x, y = C_xy
        return - ε * ( g[None,:] - squared_distances(x,y)/ε ).logsumexp(1).view(-1)
    return softmin

def lse_tensorized():
    def lse(ε,C_xy,g):
        x,y = C_xy
        return ( g[None,:] - squared_distances(x,y)/ε ).logsumexp(1).view(-1)
    return lse

def marginal_tensorized():
    #print("Tensorized version")
    def marginal(ε, C_xy, u, v):
        x, y = C_xy
        marginal_i = torch.exp( (u[:,None] + v[None,:] - squared_distances(x,y))/ε ).sum(dim = 1)
        return marginal_i
    return marginal

# ==============================================================================
#                          backend == "online"
# ==============================================================================

cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y))",
}

def softmin_online(ε, C_xy, f_y, log_conv=None):
    x, y = C_xy
    # KeOps is pretty picky on the input shapes...
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)

def lse_online(ε,C_xy, f_y, log_conv=None):
    x, y = C_xy
    # KeOps is pretty picky on the input shapes...
    return log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)

def marginal_online(ε, C_xy, b_x, a_y, log_conv=None):
    x,y = C_xy
    return log_conv( torch.Tensor([1/ε]).type_as(x), x, y, b_x.view(-1,1), a_y.view(-1,1) )


def keops_OT_plan(D, dtype="float32"):
        
    OT_plan = Genred('Exp( (F_i + G_j - SqDist(X_i,Y_j)) * E )', # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                       ['E = Pm(1)',          # First arg  is a parameter,    of dim 1
                        'X_i = Vi({})'.format(D),          # Second arg is indexed by "i", of dim 3
                        'Y_j = Vj({})'.format(D),          # Third arg  is indexed by "j", of dim 3
                        'F_i = Vi(1)',  # 4th arg: one scalar value per line
                        'G_j = Vj(1)'],         # Fourth arg is indexed by "j", of dim 2
                       reduction_op='Sum',
                       axis=1)                # Summation over "j"

    return OT_plan

def keops_lse(cost, D, dtype="float32"):
    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )",
                                 "A = Vi(1)",
                                 "X = Vi({})".format(D),
                                 "Y = Vj({})".format(D),
                                 "B = Vj(1)",
                                 "P = Pm(1)",
                                 dtype = dtype)
    return log_conv



def loss_sinkhorn_online(dim, p=2, eps =.05, Nits_max=None, acceleration = False, keops_available = True):
    
    cost = cost_formulas[p]
    
    if keops_available:
        softmin = partial( softmin_online, log_conv = keops_lse(cost, dim, dtype="float32") ) 
        marg_x = partial( marginal_online, log_conv = keops_OT_plan(dim, dtype="float32" ) )
        lse_ = partial( lse_online, log_conv=keops_lse(cost,dim,dtype="float32"))

    else:
        softmin = softmin_tensorized()
        marg_x = marginal_tensorized()
        lse_ = lse_tensorized()

    def loss(α, x, β, y):
        # The "cost matrices" are implicitely encoded in the point clouds,
        # and re-computed on-the-fly
        C_xy, C_yx = ( (x, y.detach()), (y, x.detach()) )
        u, v = sinkhorn_loop(softmin, marg_x, log_weights(α), log_weights(β),C_xy, C_yx, eps, Nits_max = Nits_max)
        return u,v

    return loss



#########################
### OT map estimators ###
#########################

class pykeops_drift():
    def __init__(self,data,pot,eps):
        self.data = data
        self.pot = pot
        self.eps = eps

    def estimator(self,x,t):
    #def T_epsn(x,yz,v,eps):
        M = squared_distances(x.T,self.data)/t
        K = -M/self.eps + self.pot/self.eps
        gammaz = -torch.max(K,dim=1)[0]
        K_shift = K + gammaz.reshape(-1,1)
        exp_ = torch.exp(K_shift)
        top_ = torch.matmul(exp_ ,self.data)
        bot_ = exp_.sum(axis=1)
        entmap = top_.T/bot_
        return (-x + entmap)/(t)

    def __call__(self,x,t):
        return self.estimator(x,t)

def T_epsn(x,yz,v,eps):
    M = squared_distances(x,yz)
    K = -M/eps + v/eps
    gammaz = -torch.max(K,dim=1)[0]
    K_shift = K + gammaz.reshape(-1,1)
    exp_ = torch.exp(K_shift)
    top_ = torch.matmul(exp_ ,yz)
    bot_ = exp_.sum(axis=1)
    entmap = top_.T/bot_
    return entmap


def euler_scheme_sde(drift, xinit, tau, steps, save_iters=False):
    dim = xinit.shape[0]
    eps = drift.eps
    steps = int(steps)
    dt = tau / steps
    k = 0
    t = 0.0

    x = xinit.clone()
    if save_iters:
        xiters = torch.zeros((steps + 1, *xinit.shape))
        xiters[0] = x
        while k < steps:
            bteps_x = drift(x, 1 - t)
            eta = torch.randn((dim, xinit.shape[1]), dtype=xinit.dtype, device=xinit.device)
            x += (dt * bteps_x) + torch.sqrt(dt * eps) * eta
            k += 1
            t = k * dt
            xiters[k] = x
        return xiters
    else:
        while k < steps:
            bteps_x = drift(x, 1 - t)
            eta = torch.randn((dim, xinit.shape[1]), dtype=xinit.dtype, device=xinit.device)
            x += (dt * bteps_x) + torch.sqrt(dt * eps) * eta
            k += 1
            t = k * dt
        return x
    
