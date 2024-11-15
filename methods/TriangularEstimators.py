import ot
import jax
import jax.numpy as jnp
import numpy as np
import ott
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

### neural OT from OTT
from ott.neural.methods import neuraldual
from ott.neural.networks import potentials
from ott.neural.networks import icnn
from ott.datasets import Dataset
import dataclasses
from scipy.stats import gaussian_kde  as orig_kde
import ott.utils as utils

import optax
from jax import random, grad, vmap
from jax.scipy.optimize import minimize
import time

from typing import Iterator, Literal, NamedTuple, Optional, Tuple

class ScaledCostOT:
    def __init__(self,dx1,dx2,beta,estimator,**args):
        self.dx1  = dx1
        self.dx2  = dx2
        self.beta = beta
        if estimator == 'NN':
            self.solver = NNTransport(**args)
        elif estimator == 'OTT':
            self.solver = EntropicTransport_OTT(**args)
        elif estimator == 'ICNN':
            self.solver = ICNNTransport(input_dim=self.dx1+self.dx2, **args)
        else:
            raise ValueError('Estimator is not yet implemented')

    def beta_vect(self):
        beta_vect = np.ones((self.dx1+self.dx2))
        beta_vect[self.dx1:] *= self.beta
        return beta_vect

    def scale_data(self, x):
        beta_vect = self.beta_vect()
        assert x.shape[1] == len(beta_vect)
        return x * np.sqrt(beta_vect)[None,:]

    def unscale_data(self, z):
        beta_vect = self.beta_vect()
        assert z.shape[1] == len(beta_vect)
        return z * np.sqrt(1./beta_vect)[None,:]

    def fit(self, source, target, **args):
        # check inputs
        assert(source.shape[1] == (self.dx1 + self.dx2))
        assert(source.shape[1] == target.shape[1])
        # scale data
        source_scaled = self.scale_data(source)
        target_scaled = self.scale_data(target)
        # solve transport problem and save potentials/map
        self.solver.estimate(source_scaled, target_scaled, **args)

    def evaluate(self, source_new, **args):
        # check inputs
        assert(source_new.shape[1] == (self.dx1 + self.dx2))
        # scale data
        source_new_scaled = self.scale_data(source_new)
        out = self.solver.evaluate_map(source_new_scaled,**args)
        return self.unscale_data(out)
    
    def inverse(self, target_new, **args):
        # check inputs
        assert(target_new.shape[1] == (self.dx1 + self.dx2))
        # scale data
        x0 = self.scale_data(np.random.randn(target_new.shape[0], target_new.shape[1]))
        target_new_scaled = self.scale_data(target_new)
        out = self.solver.inverse_map(target_new_scaled,x0,**args)
        return self.unscale_data(out)
    
    def evaluate_bridge(self, source_new, **args):
        assert(source_new.shape[1] == (self.dx1 + self.dx2))
        # scale data
        source_new_scaled = self.scale_data(source_new)
        out = self.solver.evaluate_bridge(source_new_scaled,**args)
        return self.unscale_data(out)

class EntropicTransport_OTT:
    def __init__(self, eps):
        self.eps = eps

    def estimate(self, source, target, tol=1e-3, max_iter=5000):
        # check inputs
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target must have same dimensionality")
        # define geometry
        geom_data = pointcloud.PointCloud(source, target, epsilon=self.eps)
        # solve EOT and save potentials
        out = linear.solve(geom_data, threshold=tol, max_iterations=max_iter)
        self.potentials = out.to_dual_potentials()
        self.gYjs = out.potentials[1]
        self.target = target

    def evaluate_map(self, source_new):
        return self.potentials.transport(source_new)
            
class NNTransport:
    def __init__(self, maxiters=1000000):
        self.maxiters = maxiters
        # initialize empty attributes for data and map
        self.source = None
        self.target = None
        self.G0     = None

    def estimate(self,source,target,a=None,b=None):
        n = source.shape[0]
        if a == None:
            a = np.ones(n,)/n
        if b == None:
            b = np.ones(n,)/n
        M = ot.dist(source, target, metric='sqeuclidean')
        self.G0 = ot.emd(a,b,M,numItermax=self.maxiters)
        self.source = source
        self.target = target
    
    def evaluate_map(self, source_new, algo='brute'):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm=algo).fit(self.source)
        _, indices = nbrs.kneighbors(source_new)

        target_1nn = np.zeros_like(source_new)
        loopG = self.LoopThruG(len(self.G0))
        for i, e in enumerate(indices):
            ind_ = loopG[int(e)]
            target_1nn[i] = self.target[ind_]

        return target_1nn
    
    def LoopThruG(self,n,thr=1e-8):
        l1 = []
        l2 = []
        mx = self.G0.max()
        for i in range(n):
            l1.append(i)
            for j in range(n):
                if self.G0[i, j] / mx > thr:
                    l2.append(j)
        return dict(zip(l1,l2))

@dataclasses.dataclass
class KDE:
    """A mixture of Gaussians.
  
    Args:
      name: the name specifying the centers of the mixture components:
  
        - ``simple`` - data clustered in one center,
        - ``circle`` - two-dimensional Gaussians arranged on a circle,
        - ``square_five`` - two-dimensional Gaussians on a square with
          one Gaussian in the center, and
        - ``square_four`` - two-dimensional Gaussians in the corners of a
          rectangle
  
      batch_size: batch size of the samples
      rng: initial PRNG key
      scale: scale of the Gaussian means
      std: the standard deviation of the individual Gaussian samples
    """
    data: np.array
    batch_size: int
    rng: jax.Array
  
    def __post_init__(self) -> None:
        self.kde = orig_kde(self.data.T, bw_method=0.00)

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
          A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()
    
    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.rng
        while True:
          samples = self.kde.resample(self.batch_size).T
          yield samples

def minibatch_samplers(
    source_data: np.array,
    target_data: np.array,
    train_batch_size: int = 256,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
    """Gaussian samplers.
    Args:
      name_source: name of the source sampler
      name_target: name of the target sampler
      train_batch_size: the training batch size
      valid_batch_size: the validation batch size
      rng: initial PRNG key
    Returns:
      The dataset and dimension of the data.
    """
    rng = utils.default_prng_key(rng)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    train_dataset = Dataset(
        source_iter=iter(
            KDE(source_data, batch_size=train_batch_size, rng=rng1)
        ),
        target_iter=iter(
            KDE(target_data, batch_size=train_batch_size, rng=rng2)
        )
    )
    # valid_dataset = Dataset(
    #     source_iter=iter(
    #         KDE(source_data, batch_size=valid_batch_size, rng=rng3)
    #     ),
    #     target_iter=iter(
    #         KDE(target_data, batch_size=valid_batch_size, rng=rng4)
    #     )
    # )
    return train_dataset #, valid_dataset

def training_callback(step, learned_potentials):
    if step % 50 == 0:
        print(step)

class ICNNTransport:
    def __init__(self, input_dim, hidden_dim=64, n_iters=1000, batch_size=128):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def estimate(self,source,target):
        
        train_dataloaders= minibatch_samplers(
                                source_data = source,
                                target_data = target,
                                train_batch_size=self.batch_size)
        
        neural_f = potentials.PotentialMLP(
            dim_hidden=[self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            is_potential=True,  # returns the gradient of the potential.  
        )
        neural_g = potentials.PotentialMLP(
            dim_hidden=[self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            is_potential=False,  # returns the gradient of the potential.
        )
        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-3, decay_steps=self.n_iters, alpha=1e-2
        )
        optimizer_f = optax.adam(learning_rate=lr_schedule, b1=0.5, b2=0.5)
        optimizer_g = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999)

        neural_dual_solver = neuraldual.W2NeuralDual(
            self.input_dim,
            neural_f,
            neural_g,
            optimizer_f,
            optimizer_g,
            back_and_forth=False,
            num_train_iters=self.n_iters,
            valid_freq=1000, 
            log_freq=1000
        )
        self.learned_potentials = neural_dual_solver(
            *train_dataloaders,
            *train_dataloaders,
            #*valid_dataloaders,
            callback=training_callback,
        )

    def evaluate_map(self,source_new):
        return self.learned_potentials.transport(source_new)
    
    def inverse_map(self,target_new):
        return self.learned_potentials.transport(target_new)

