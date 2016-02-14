"""
gaussian_process.py
-------------------
This module performs basic Gaussian process regression using the squared
exponential kernel. 

Author: Rui Shu
"""

import numpy as np
from numpy.linalg import cholesky as chol 

class GaussianProcess(object):
    def __init__(self, ls=None, amp=None, noise=None, 
                 lsr=(.1, .2, 1), ampr=(1., 2.), noiser=(.1, .2), 
                 keps=1e-9):
        """ Gaussian Process Class

        Initialization of Gaussian process hyperparameters

        Parameters
        ----------
        ls : (n_dim,) array/list of floats
            Array of length scales to be used
        amp : float
            Amplitude parameter in kernel
        noise : float
            Noise parameter in kernel
        lsr : 3-tuple (float, float, int)
            The range for random initializtion, and ndim number of ls's
        ampr : 2-tuple (float, float)
            The range of random initialization for amp 
        noiser : 2-tuple (float, float)
            The range of random initialization for noise 
        keps : float
            Small constant added to diag of kernel matrix for stability
        """
        # parameters for non-learning
        self.params = {}
        self.params['ls'] = (np.array(ls) if ls is not None
                             else np.random.uniform(*tuple(lsr)))
        self.params['amp'] = (np.array(amp) if amp is not None
                              else np.random.uniform(*tuple(ampr)))
        self.params['noise'] = (np.array(noise) if noise is not None
                                else np.random.uniform(*tuple(noiser)))
        self.params['keps'] = keps
        # self.Ki stores the inverse of the kernel
        self.Ki = None
        
    def fit(self, X, y, nepochs=1000, batch_size=None, train=True,
            learning_rate=1e-1, learning_rate_decay=1.,
            verbose=0):
        """ Feeds in input data and trains parameters of the GP if desired
        
        Parameters
        ----------
        X : (n_samples, n_dim) float matrix
            independent variables
        y : (n_samples,) float matrix
            response variable
        train : boolean
            Whether to train parameters or not
        """
        # Preprocess:
        X = np.atleast_2d(X).astype(float)
        y = y.reshape(-1, 1).astype(float)
        # Scale to unit hypercube
        y_m = y.mean()
        y_scale = y.max() - y.min()
        X_m = X.min(axis=0)
        X_scale = X.max(axis=0) - X.min()
        # Check for scale being 0
        y_scale = y_scale if y_scale > 0 else 1.
        X_scale[X_scale <= 0] = 1.
        self.transform = {'ym': y_m,
                          'ys': y_scale,
                          'Xm': X_m,
                          'Xs': X_scale}
        X = (X - X_m)/X_scale
        y = (y - y_m)/y_scale

        if train:
            self.train(X, y, nepochs, batch_size, 
                       learning_rate, learning_rate_decay,
                       verbose)
        
        K = self.K(X)
        self.Ki = np.linalg.inv(K)
        self.X = X
        self.y = y
        return self
        
    def train(self, X, y, nepochs, batch_size, learning_rate,
              learning_rate_decay, verbose):
        """ Update parameters via gradient ascend to maximize marginal log
        likelihood.
        """
        batch_size = len(X) if batch_size is None else batch_size
        if batch_size > len(X):
            string = "Batch size larger than sample size. Reducing it to. {0:d}"
            print string.format(len(X))
            batch_size = len(X)
        batch_count = 0

        # Compute the frequency with which print statements should arrive
        # I expect at least 10 epoch print outs
        
        for epoch in xrange(1, nepochs+1):
            shuffle = np.random.permutation(len(X))
            X = X[shuffle]
            y = y[shuffle]
            nbatch = len(X)/batch_size  # good enough approximation
            for i in xrange(nbatch):
                batch_count += 1
                X_batch = X[i*batch_size : (i+1)*batch_size]
                y_batch = y[i*batch_size : (i+1)*batch_size]
                L, dL = self.L(X_batch, y_batch, grad=True)
                print dL['ls']
                for key in ['ls', 'amp']:
                    self.params[key] += learning_rate * dL[key]
                    if verbose > 2:
                        print key, ":", self.params[key]
                        print key, "gradient:", dL[key]
                if batch_count % 100 == 0 and verbose > 1:
                    print "...Batch:", batch_count, "Negative log likelihood:", -L
            if epoch % (nepochs/10) == 0 and verbose > 0:
                print "Epoch:", epoch, "Negative log likelihood:", -L

        if verbose > 0:
            print "*"*80
            print "Final negative log likelihood", -L
            print "Final parameters:"
            for key in self.params:
                print key, ":", self.params[key]
            print "*"*80
                
    def L(self, X, y, grad=False):
        """ Computes the log likelihood and returns the gradient w.r.t. ls, amp,
        and noise if desired.
        """
        # Set up cache
        cache = {}
        K = self.K(X)
        Ki = np.linalg.inv(K)
        Kiy = Ki.dot(y)
        norm = len(X)
        cache['K'] = K
        cache['Ki'] = Ki
        cache['Kiy'] = Kiy
        cache['norm'] = norm
        # Compute L and dL
        L = -0.5*y.T.dot(Kiy).sum() - np.log(np.diag(chol(K))).sum()
        L -= len(X)/2 * np.log(2*np.pi)
        if grad:
            return L/norm, self.L_grad(X, y, cache)
        else:
            return L/norm

    def L_grad(self, X, y, cache=None):
        """ Computes the gradient of the log likelihood w.r.t. ls, amp, and
        noise.
        """
        if cache is None:
            K = self.K(X)
            Ki = np.linalg.inv(K)
            Kiy = Ki.dot(y)
            norm = len(X)
        else:
            K = cache['K']
            Ki = cache['Ki']
            Kiy = cache['Kiy']
            norm = cache['norm']
        pre = Kiy.dot(Kiy.T) - Ki
        dK = self.K_grad(X)
        grads = {}
        grads['ls'] = np.zeros(len(dK['ls']))
        for i in xrange(len(dK['ls'])):
            grads['ls'][i] = np.sum(np.diag(pre.dot(dK['ls'][i])))/norm/2
        grads['amp'] = np.sum(np.diag(pre.dot(dK['amp'])))/norm/2
        grads['noise'] = np.sum(np.diag(pre.dot(dK['noise'])))/norm/2
        return grads
    
    def K(self, X, grad=False):
        """ Computes the covariance matrix using X and returns the gradient
        w.r.t. ls, amp, and noise if desired.
        """
        keps = self.params['keps']
        noise = self.params['noise']
        K = (
            self.compute_kernel(X) +
            (keps + noise**2)*np.eye(len(X))
        )
        if grad:
            return K, self.K_grad(X)
        else:
            return K
        
    def K_grad(self, X):
        """ Computes the gradient of the covariance matrix element-wise
        w.r.t. ls, amp, and noise.
        """
        ls = self.params['ls']
        amp = self.params['amp']
        noise = self.params['noise']
        grads = {}
        # Compute dK w.r.t. noise
        grads['noise'] = 2*noise*np.eye(len(X))
        # Compute dK w.r.t. amp
        sqdist = self.compute_sqdist(X/ls)
        grads['amp'] = 2*amp * np.exp(-0.5*sqdist)
        # Compute dK w.r.t. ls
        grads['ls'] = []
        precompute = amp**2 * np.exp(-0.5*sqdist)
        for i in xrange(X.shape[1]):
            val = precompute/ls[i]**3
            val *= self.compute_sqdist(X[:, i, None])
            grads['ls'].append(val)
        grads['ls'] = np.array(grads['ls'])
        return grads

    def compute_kernel(self, X, Y=None):
        """ Compue the kernel matrix for X x Y, keeping in mind the scaling
        w.r.t. ls
        """
        ls = self.params['ls']
        amp = self.params['amp']
        noise = self.params['noise']
        if Y is None:
            sqdist = self.compute_sqdist(X/ls)
        else:
            sqdist = self.compute_sqdist(X/ls, Y/ls)
        return amp**2 * np.exp(-0.5*sqdist)

    @staticmethod
    def compute_sqdist(X, Y=None):
        """ Compute the square distance between the rows of X v. Y
        """
        if Y is None:
            a = (X**2).sum(axis=1).reshape(-1, 1)
            b = a.T
            c = X.dot(X.T)
        else:
            a = (X**2).sum(axis=1).reshape(-1, 1)
            b = (Y**2).sum(axis=1)
            c = X.dot(Y.T)
        return a + b - 2*c

    def predict(self, X_):
        """ Predict w.r.t. X_
        """
        # Push X_ into hypercube
        X_ = (X_ - self.transform['Xm'])/self.transform['Xs']
        K_ = self.compute_kernel(self.X, X_)
        y_ = K_.T.dot(self.Ki).dot(self.y)
        # Compute variance
        var = self.params['amp']**2 - (K_.T * K_.T.dot(self.Ki)).sum(axis=1)
        var = (var * self.transform['ys']**2).reshape(-1, 1)
        # Convert y_ back to original input space
        y_ = y_ * self.transform['ys'] + self.transform['ym']
        return y_, var

def main():
    import matplotlib.pyplot as plt
    # make some random data
    # np.random.seed(45)
    n = 10
    X = np.random.uniform(0., 1, (n, 1))
    y = np.squeeze(np.sin(X*10)) + np.random.randn(len(X))*.1

    gp = GaussianProcess(ls=[.1],
                         amp=.1,
                         noise=2e-2)
    gp.fit(X, y, nepochs=100, batch_size=1000,
           learning_rate=1e-2, learning_rate_decay=.999,
           verbose=1)
    X_ = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_, var = gp.predict(X_)
    plt.scatter(X, y, c='r')
    plt.plot(X_, y_ + var**.5, c='g')
    plt.plot(X_, y_ - var**.5, c='g')
    plt.plot(X_, y_, c='y', linewidth=2)
    plt.show()
    
if __name__ == "__main__":
    main()
                    
        
