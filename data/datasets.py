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
        lik = self.conditional_pdf(x,y)
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

if __name__=='__main__':

    pi = tanh_v2()
    Xs = pi.sample_joint(5000)

    # define grid
    Ng = 100
    xx = np.linspace(-3,3,Ng)
    yy = np.linspace(-0.8,0.8,Ng)
    [Xg, Yg] = np.meshgrid(xx, yy)
    
    # evaluate density
    pdf_g = pi.joint_pdf(Xg.flatten()[:,None], Yg.flatten()[:,None])
    pdf_g = pdf_g.reshape((Ng, Ng))

    import matplotlib.pyplot as plt

    # plot joint density
    plt.figure()
    plt.contourf(Xg, Yg, pdf_g)
    plt.plot(Xs[:,0], Xs[:,1], '.r')
    plt.xlim(-3,3)
    plt.ylim(-0.8,0.8)
    plt.colorbar()
    plt.show()

    # evaluate conditional density
    yst = 0.5
    pdf_cond = pi.conditional_pdf(np.linspace(-3,3,1000)[:,None], np.tile(yst,(1000,1)))
    cond_samples = pi.sample_conditional(yst, 1000)

    plt.figure()
    plt.plot(np.linspace(-3,3,1000), pdf_cond, '-b')
    plt.hist(cond_samples, bins=30, density=True)
    plt.show()

