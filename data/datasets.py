import numpy as np
import scipy.stats as stats
from scipy.integrate import odeint
import sklearn.datasets

####  2-Dimensional Gaussian ####      

class TwoDimGaussian():
    def __init__(self,mu1,mu2,sigma1,sigma2,gamma):
        self.mu1    = mu1
        self.mu2    = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gamma  = gamma

    def sample_joint(self,size):
        # define joint mean and covariance
        s1 = self.sigma1**2
        s2 = self.gamma*self.sigma1*self.sigma2
        s3 = self.sigma2**2
        cov_matrix = np.matrix([[s1,s2],[s2,s3]])
        mu_vector = np.array([self.mu1,self.mu2])
        # draw samples
        return stats.multivariate_normal.rvs(mu_vector,cov_matrix,size=size)

    def conditional_mean(self, fixed_x1): # X2|X1
        return self.mu2 + (self.sigma2/self.sigma1) * self.gamma * (fixed_x1 - self.mu1)

    def conditional_cov(self): # X2|X1
        return (1-self.gamma**2)*(self.sigma2**2)

    def sample_conditional(self, fixed_x1, size):
        cond_mean = self.conditional_mean(fixed_x1)
        cond_std  = np.sqrt(self.conditional_cov())
        return stats.norm.rvs(loc=cond_mean, scale=cond_std, size=size)

    def conditional_pdf(self, fixed_x1, x2):
        cond_mean = self.conditional_mean(fixed_x1)
        cond_std  = np.sqrt(self.conditional_cov())
        return stats.norm.pdf(x2, loc=cond_mean, scale=cond_std)

####  N-Dimensional Gaussian ####

class MultivariateGaussian():
    def __init__(self, mu1, mu2, Sigma11, Sigma22, Sigma21):
        self.dim_x1  = len(mu1)
        self.dim_x2  = len(mu2)
        # check inputs
        assert(np.all(Sigma11.shape == (self.dim_x1,self.dim_x1)))
        assert(np.all(Sigma22.shape == (self.dim_x2,self.dim_x2)))
        assert(np.all(Sigma21.shape == (self.dim_x2,self.dim_x1)))
        # save inputs
        self.mu1     = mu1
        self.mu2     = mu2
        self.Sigma11 = Sigma11
        self.Sigma22 = Sigma22
        self.Sigma21 = Sigma21

    def sample_joint(self, N):
        # define joint mean and covariance
        mu_vector = np.hstack([self.mu1, self.mu2])
        cov_matrix = np.vstack((np.hstack([self.Sigma11, self.Sigma21.T]),np.hstack([self.Sigma21, self.Sigma22])))
        # generate samples
        ndim_gaussian = stats.multivariate_normal.rvs(mu_vector, cov_matrix, N)
        return ndim_gaussian
        
    def conditional_mean(self, fixed_x1): # X2|X1
        assert(len(fixed_x1) == self.dim_x1)
        return self.mu2 + np.dot(np.linalg.solve(self.Sigma11, self.Sigma21.T).T,(fixed_x1 - self.mu1)[:,None])[:,0]

    def conditional_cov(self): # X2|X1
        return self.Sigma22 - np.dot(self.Sigma21, np.linalg.solve(self.Sigma11, self.Sigma21.T))
    
    def sample_conditional(self, fixed_x1, N):
        assert(len(fixed_x1) == self.dim_x1)
        cond_mean = self.conditional_mean(fixed_x1)
        cond_var  = self.conditional_cov()
        return stats.multivariate_normal.rvs(cond_mean, cond_var, N)

    def conditional_pdf(self, fixed_x1, x2):
        assert(len(fixed_x1) == self.dim_x1)
        assert(x2.shape[1] == self.dim_x2)
        cond_mean = self.conditional_mean(fixed_x1)
        cond_var  = self.conditional_cov()
        return stats.multivariate_normal.pdf(x2, cond_mean, cond_var)

#### Banana Dataset #### 

class Banana():
    def __init__(self, reverse=False):
        self.x1_mean = 0
        self.x1_std  = 1
        self.x2_mean = -1
        self.x2_std  = 1
        self.reverse = reverse
          
    def sample_joint(self, N):
        # draw samples
        x1 = self.x1_std * np.random.randn(N,1) + self.x1_mean
        x2 = self.x2_std * np.random.randn(N,1) + x1**2 + self.x2_mean
        # flip ordering of x1 and x2
        if self.reverse==True:
            x = np.hstack((x2,x1))
        else:
            x = np.hstack((x1,x2))
        return x

    def joint_pdf(self, x):
        if self.reverse == True:
            x = np.hstack((x[:,1].reshape(-1,1),x[:,0].reshape(-1,1)))
        pdf_x1   = stats.norm.pdf(x[:,0], self.x1_mean, self.x1_std)
        pdf_cond = stats.norm.pdf(x[:,1], x[:,0]**2 + self.x2_mean, self.x2_std)
        return pdf_x1 * pdf_cond

    def conditional_pdf(self, fixed_x1, x2):
        if self.reverse == True:
            raise ValueError('Reverse doesn''t have closed form conditional')
        conditional_mean = fixed_x1**2 + self.x2_mean 
        pdf = stats.norm.pdf(x2, conditional_mean, self.x2_std)
        return pdf

    def sample_conditional(self, fixed_x1, N):
        if self.reverse == True:
            raise ValueError('Reverse doesn''t have closed form conditional')
        conditional_mean = fixed_x1**2 + self.x2_mean 
        return stats.norm(conditional_mean, self.x2_std).rvs(size=N)

#### Tanh Dataset #### 

class base_tanh():
    def __init__(self):
        self.x_a = -3
        self.x_b = 3
    
    def sample_prior(self,N):
        return (self.x_b - self.x_a) * np.random.rand(N,1) + self.x_a
       
    def sample_joint(self, N):
        x = self.sample_prior(N)
        y = self.sample_data(x)
        data = np.hstack((x,y))
        return data
    
    def prior_pdf(self,x):
        supp = np.ones((x.shape[0],1))
        supp[x[:,0]< self.x_a] = 0
        supp[x[:,0]>self.x_b] = 0
        pi = 1./(self.x_b-self.x_a) * supp
        return pi
    
    def joint_pdf(self,x,y):
        prior = self.prior_pdf(x)
        lik = self.likelihood_function(x,y)
        return prior * lik


class tanh_v1(base_tanh):
    # y = tanh(x) + Gamma[1,0.3]
    def __init__(self):
        super(tanh_v1, self).__init__()
        self.y_alpha = 1.
        self.y_beta = 1./0.3

    def sample_data(self,x):
        N = x.shape[0]
        g = stats.gamma.rvs(self.y_alpha,loc=0,scale= 1./self.y_beta,size =(N,1))
        return np.tanh(x) + g 

    def sample_conditional(self,fixed_x,N):
        x = np.repeat(fixed_x,N)
        gamma_sample = stats.gamma.rvs(self.y_alpha,loc=0,scale=1/self.y_beta,size=N)
        y_given_x = np.tanh(x) + gamma_sample 
        data = y_given_x
        return data
    
    def conditional_pdf(self,x,y):
        # evaluate derivative of inverse map
        derTinv = 1.
        # evaluate inverse map under reference density
        g = y-np.tanh(x)
        return stats.gamma.pdf(g,self.y_alpha,loc=0,scale = 1./self.y_beta) * np.abs(derTinv)
  

class tanh_v2(base_tanh):
    # y = tanh(x + n), n ~ Normal(0,0.05)
    def __init__(self):
        super(tanh_v2, self).__init__()
        self.n_mean = 0.
        self.n_std = np.sqrt(0.05)
    
    def sample_data(self,x):
        N = x.shape[0]
        n = stats.norm.rvs(loc=self.n_mean,scale=self.n_std,size=(N,1))
        return np.tanh(x+n)
    
    def sample_conditional(self,fixed_x,N):
        x = np.repeat(fixed_x,N)
        norm_sample = stats.norm.rvs(loc=self.n_mean,scale=self.n_std,size=(N,))
        y_given_x = np.tanh(x + norm_sample) 
        data = y_given_x
        return data
    
    def conditional_pdf(self,x,y):
        # evaluate derivative of inverse map
        derTinv = (1-y**2)**(-1)
        # evaluate inverse map under reference density
        n = np.arctanh(y) - x
        lik = stats.norm.pdf(n,loc=self.n_mean,scale=self.n_std) * np.abs(derTinv)
        lik[np.isnan(lik)] = 0.
        return lik
    
    def map(self,x1,z):
        # evaluate map pushing forward N(0,1) to p(x2|x1)
        return np.tanh(x1 + self.n_std*z)
    
 
class tanh_v3(base_tanh):
    # y = gamma*tanh(x), gamma ~ Gamma(1,0.3)
    def __init__(self):
        super(tanh_v3, self).__init__()
        self.y_alpha = 1.
        self.y_beta = 1./0.3
    
    def sample_data(self,x):
        N = x.shape[0]
        g = stats.gamma.rvs(self.y_alpha, loc =0 ,scale=1./self.y_beta,size =(N,1))
        return np.tanh(x) * g
    
    def sample_conditional(self,fixed_x,N):
        x = np.repeat(fixed_x,N)
        gamma_sample = stats.gamma.rvs(self.y_alpha,loc=0,scale=1./self.y_beta,size=N)
        y_given_x = np.tanh(x) * gamma_sample 
        data = y_given_x
        return data
    
    def conditional_pdf(self,x,y):
        # evaluate derivative of inverse map
        derTinv = 1./np.tanh(x)
        # evaluate inverse map under reference density
        g = y /np.tanh(x)
        return stats.gamma.pdf(g,self.y_alpha,loc=0,scale=1./self.y_beta) * np.abs(derTinv)

#### Lotka-Volterra ####

class Deterministic_lotka_volterra():
    def __init__(self,T):
        self.d = 4
        self.alpha_mu = -0.125
        self.alpha_std = 0.5
        self.beta_mu = -3
        self.beta_std = 0.5
        self.gamma_mu = -0.125
        self.gamma_std = 0.5
        self.delta_mu = -3
        self.delta_std = 0.5
        self.x0 = [30,1]
        self.T = T
        self.obs_std = np.sqrt(0.1)
    
    def sample_prior(self,N):
        alpha = stats.lognorm.rvs(scale = np.exp(self.alpha_mu),s=self.alpha_std,size=(N,))
        beta = stats.lognorm.rvs(scale=np.exp(self.beta_mu),s=self.beta_std,size = (N,))
        gamma = stats.lognorm.rvs(scale=np.exp(self.gamma_mu),s=self.gamma_std,size = (N,))
        delta = stats.lognorm.rvs(scale=np.exp(self.delta_mu),s=self.delta_std,size = (N,))
        return np.vstack((alpha,beta,gamma,delta)).T
    
    def ode_rhs(self,z,t,theta):
        alpha,beta,gamma,delta = theta
        fz1 = alpha * z[0] - beta * z[0]*z[1]
        fz2 = -gamma * z[1] + delta * z[0]*z[1]
        return np.array([fz1,fz2])
    
    def simulate_ode(self,theta,tt):
        assert(theta.size == self.d)
        return odeint(self.ode_rhs,self.x0,tt,args =(theta,))
    
    def sample_data(self,theta):
        if len(theta.shape) == 1: 
            theta = theta[np.newaxis,:]
        assert(theta.shape[1] == self.d)
        tt = np.arange(0,self.T,step=2)
        nt = 2*(len(tt)-1)
        xt = np.zeros((theta.shape[0],nt))
        for j in range(theta.shape[0]):
            yobs = self.simulate_ode(theta[j,:],tt)
            yobs = np.abs(yobs[1:,:]).ravel()
            xt[j,:] = np.array([stats.lognorm.rvs(scale =x,s=self.obs_std) for x in yobs])
        # return (xt,tt)
        return xt

    def sample_joint(self,N):
        x = self.sample_prior(N)
        y = self.sample_data(x)
        xy = np.concatenate((x,y),-1) ### FIX LATER
        return xy
    
    def log_prior_pdf(self,theta):
        assert(theta.shape[1]==self.d)
        prior_mean = [self.alpha_mu,self.beta_mu,self.gamma_mu,self.delta_mu]
        prior_std = [self.alpha_std,self.beta_std,self.gamma_std,self.delta_std]
        return np.sum(stats.lognorm.logpdf(theta,scale=np.exp(prior_mean),s=prior_std))
    
    def log_likelihood(self,theta,yobs):
        assert(theta.shape[1]==self.d)
        assert(yobs.size == (self.T-2))
        tt = np.arange(0,self.T,step=2)
        loglik = np.zeros(theta.shape[0])
        for j in range(theta.shape[0]):
            xt = self.simulate_ode(theta[j,:],tt)
            xt = np.abs(xt[1:,:]).ravel()
            loglik[j] = np.sum([stats.lognorm.logpdf(yobs,scale=xt,s=self.obs_std)])
        return loglik

#### 2D Toy Problems ####

class ToyDatasets():
    def __init__(self, dataset, rng=None):
        self.dataset = dataset
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

class Checkerboard(ToyDatasets):
    def __init__(self, scale=0.45, rng=None):
        super(Checkerboard, self).__init__(rng)
        self.scale  = scale
        self.domain = [-2,2]

    def sample_joint(self, N):
        dom_range = self.domain[1] - self.domain[0]
        x1  = np.random.rand(N)*dom_range + self.domain[0]
        x2_ = np.random.rand(N) - np.random.randint(0,2,N)*2
        x2  = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:,None],x2[:,None]],1) / self.scale
        return data

    def joint_pdf(self, x):
        # re-scale data
        x *= self.scale
        # evaluate marginal on x1
        supp_x1 = ((x[:,0] >= self.domain[0]) & (x[:,0] < self.domain[1]))
        dom_range = self.domain[1] - self.domain[0]
        pdf_x1 = 1./dom_range * supp_x1 * self.scale
        # evaluate conditional of x2|x1
        x2_norm = x[:,1] - (np.floor(x[:,0]) % 2)
        supp_x2 = ((x2_norm >= -2) & (x2_norm < -1)) | ((x2_norm >= 0) & (x2_norm < 1))
        pdf_x2condx1 = 1./2. * supp_x2 * self.scale
        # evaluate joint pdf
        joint_pdf = pdf_x1 * pdf_x2condx1
        return joint_pdf

    def conditional_pdf(self, x2, x1):
        # re-scale data
        x = np.concatenate((x1,x2),axis=1)
        x *= self.scale
        # evaluate conditional of x2|x1
        x2_norm = x[:,1] - (np.floor(x[:,0]) % 2)
        supp_x2 = ((x2_norm >= -2) & (x2_norm < -1)) | ((x2_norm >= 0) & (x2_norm < 1))
        pdf_x2condx1 = 1./2. * supp_x2 * self.scale
        return pdf_x2condx1

class GaussianMixture(ToyDatasets):
    def __init__(self, scale=3., rng=None):
        super(GaussianMixture, self).__init__(rng)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
                (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        self.centers = [(scale * x, scale * y) for x, y in centers]
        self.std     = 0.4

    def sample_joint(self, N):
        dataset = []
        for i in range(N):
            point = self.rng.randn(2) * self.std
            idx = self.rng.randint(8)
            center = self.centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        return dataset

    def joint_mean(self):
        mean = np.zeros((2,))
        n_centers = len(self.centers)
        for i in range(n_centers):
            mean += np.array(self.centers[i]) / n_centers
        return mean

    def joint_cov(self):
        cov = np.zeros((2,2))
        n_centers = len(self.centers)
        global_mean = self.joint_mean()        
        for i in range(n_centers):
            center_mean   = self.centers[i]
            center_var    = self.std**2*np.eye(2) 
            center_weight = 1/len(self.centers)
            cov += center_weight * (center_var + np.dot(center_mean - global_mean, center_mean - global_mean))
        return cov

    def joint_pdf(self, x):
        pdf = np.zeros(x.shape[0])
        for i in range(len(self.centers)):
            center_mean   = self.centers[i]
            center_cov    = self.std**2*np.eye(2) 
            center_weight = 1/len(self.centers)
            pdf += center_weight * stats.multivariate_normal.pdf(x, center_mean, center_cov)
        return pdf

    def conditional_pdf(self, x, y):
        # evaluate joint mean & covariance
        joint_mean = self.joint_mean()
        joint_var  = self.joint_cov()
        # evaluate conditional PDF
        pdf = np.zeros(x.shape[0])
        for i in range(len(self.centers)):
            cond_mean   = joint_mean[0] + joint_var[0,1]/joint_var[1,1]*(y - joint_mean[1])
            center_cov  = joint_var[0,0] - joint_var[0,1]/joint_var[1,1] * joint_var[1,0]
            print(cond_mean)
            print(center_cov)
            cond_weight = 1 ### FIX
            pdf += cond_weight * stats.multivariate_normal.pdf(x - cond_mean, np.zeros(1,), center_cov)
        return pdf

def PinWheel(ToyDatasets):
    def __init__(self):

        self.radial_std = 0.3
        self.tangential_std = 0.1
        self.num_classes = 5
        self.rate = 0.25

    def sample_joint(self, N):

        # sample radians
        rads = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)

        # sample labels
        num_per_class = N // self.num_classes
        features = self.rng.randn(self.num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * self.rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    def joint_pdf(self, N):
        raise ValueError('not implemented')

if __name__=='__main__':

    pi = GaussianMixture()
    Xs = pi.sample_joint(5000)

    # define grid
    Ng = 100
    xx = np.linspace(-4.25,4.25,Ng)
    [Xg, Yg] = np.meshgrid(xx, xx)
    
    # evaluate density
    pdf_g = pi.joint_pdf(np.vstack((Xg.flatten(), Yg.flatten())).T)
    pdf_g = pdf_g.reshape((Ng, Ng))

    import matplotlib.pyplot as plt

    # plot joint density
    plt.figure()
    plt.contourf(Xg, Yg, pdf_g)
    plt.plot(Xs[:,0], Xs[:,1], '.r')
    plt.colorbar()
    plt.show()

    # evaluate conditional density
    yst = 1.0
    pdf_cond = pi.conditional_pdf(xx[:,None], np.tile(yst, (Ng,1)))

    # extract approximate conditional samples
    idx = np.where(np.abs(Xs[:,1] - yst) < 1e-1)

    plt.figure()
    plt.plot(xx, pdf_cond, '-b')
    plt.hist(Xs[idx[0],0], bins=30, density=True)
    plt.show()

