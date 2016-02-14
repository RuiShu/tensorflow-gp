import numpy as np

def check_grad(f, x, h=1e-4, verbose=0):
    """ checks whether the computed gradient is correct
    
    Parameters
    ----------
    f : lambda function
        Takes in a particular argument x and computes the loss and gradient 
        w.r.t. x
    x : np.array
        A parameter (or array of parameters) to be passed into f
    h : float
        The discrete step used for central difference 
    """
    max_dev = -np.inf
    if type(x) == np.ndarray and x.shape is not ():
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        _, ana_grad = f(x)
        num_grad = np.zeros_like(ana_grad)
        while not it.finished:
            ix = it.multi_index
            x[ix] += h/2
            fxph, _ = f(x) 
            x[ix] -= h
            fxmh, _ = f(x) 
            x[ix] += h/2
            num_grad[ix] = (fxph - fxmh) / h
            if verbose > 1:
                print "Numeric:", num_grad[ix]
                print "Analytic:", ana_grad[ix]
            if verbose > 0:
                diff = ana_grad[ix] - num_grad[ix]
                if abs(diff).max() > max_dev:
                    max_dev = abs(diff).max()
                print "Numerical v. analytical gradient difference:\n", diff
            it.iternext()
    elif type(x) == np.ndarray and x.shape is ():
        _, ana_grad = f(x)
        original = x
        x += h/2
        fxph, _ = f(x)
        x -= h
        fxmh, _ = f(x)
        x += h/2
        num_grad = (fxph - fxmh) / h
        if verbose > 1:
            print "Numeric:", num_grad
            print "Analytic:", ana_grad
        if verbose > 0:
            diff = ana_grad - num_grad
            if abs(diff).max() > max_dev:
                max_dev = abs(diff).max()
            print "Numerical v. analytical gradient difference:\n", diff
    if verbose:
        print "Max deviation:", max_dev
    return ana_grad, num_grad

if __name__ == "__main__":
    f = lambda x: (x.sum(), np.ones_like(x))
    check_grad(f, x=np.random.randn(3, 3), verbose=True)
