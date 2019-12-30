#!/usr/bin/env python
"""

As part of the :code:`bilby.result.Result` object, we provide a method to
calculate the Occam factor (c.f., Chapter 28, `Mackay "Information Theory,
Inference, and Learning Algorithms"
<http://www.inference.org.uk/itprnn/book.html>`). This is an approximate
estimate based on the posterior samples, and assumes the posteriors are well
approximate by a Gaussian.

The Occam factor penalizes models with larger numbers of parameters (or
equivalently a larger "prior volume"). This example won't try to go through
explaining the meaning of this, or how it is calculated (those details are
sufficiently well done in Mackay's book linked above). Insetad, it demonstrates
how to calculate the Occam factor in :code:`bilby` and shows an example of it
working in practise.

If you have a :code:`result` object, the Occam factor can be calculated simply
from :code:`result.occam_factor(priors)` where :code:`priors` is the dictionary
of priors used during the model fitting. These priors should be uniform
priors only. Other priors may cause unexpected behaviour.

In the example, we generate a data set which contains Gaussian noise added to a
quadratic function. We then fit polynomials of differing degree. The final plot
shows that the largest evidence favours the quadratic polynomial (as expected)
and as the degree of polynomial increases, the evidence falls of in line with
the increasing (negative) Occam factor.

Note - the code uses a course 100-point estimation for speed, results can be
improved by increasing this to say 500 or 1000.

"""
from __future__ import division
import bilby
import numpy as np
import matplotlib.pyplot as plt

# A few simple setup steps
label = 'occam_factor'
outdir = 'outdir'
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

sigma = 1

N = 100
time = np.linspace(0, 1, N)
coeffs = [1, 2, 3]
data = np.polyval(coeffs, time) + np.random.normal(0, sigma, N)

fig, ax = plt.subplots()
ax.plot(time, data, 'o', label='data', color='C0')
ax.plot(time, np.polyval(coeffs, time), label='true signal', color='C1')
ax.set_xlabel('time')
ax.set_ylabel('y')
ax.legend()
fig.savefig('{}/{}_data.png'.format(outdir, label))


class Polynomial(bilby.Likelihood):
    def __init__(self, x, y, sigma, n):
        """
        A Gaussian likelihood for polynomial of degree `n`.

        Parameters
        ----------
        x, y: array_like
            The data to analyse.
        sigma: float
            The standard deviation of the noise.
        n: int
            The degree of the polynomial to fit.
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.N = len(x)
        self.n = n
        self.keys = ['c{}'.format(k) for k in range(n)]
        self.parameters = {k: None for k in self.keys}

    def polynomial(self, x, parameters):
        coeffs = [parameters[k] for k in self.keys]
        return np.polyval(coeffs, x)

    def log_likelihood(self):
        res = self.y - self.polynomial(self.x, self.parameters)
        return -0.5 * (np.sum((res / self.sigma)**2) +
                       self.N * np.log(2 * np.pi * self.sigma**2))


def fit(n):
    likelihood = Polynomial(time, data, sigma, n)
    priors = {}
    for i in range(n):
        k = 'c{}'.format(i)
        priors[k] = bilby.core.prior.Uniform(0, 10, k)

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, npoints=100, outdir=outdir,
        label=label)
    return (result.log_evidence, result.log_evidence_err,
            np.log(result.occam_factor(priors)))


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

log_evidences = []
log_evidences_err = []
log_occam_factors = []
ns = range(1, 11)
for l in ns:
    e, e_err, o = fit(l)
    log_evidences.append(e)
    log_evidences_err.append(e_err)
    log_occam_factors.append(o)

ax1.errorbar(ns, log_evidences, yerr=log_evidences_err,
             fmt='-o', color='C0')
ax1.set_ylabel("Unnormalized log evidence", color='C0')
ax1.tick_params('y', colors='C0')

ax2.plot(ns, log_occam_factors,
         '-o', color='C1', alpha=0.5)
ax2.tick_params('y', colors='C1')
ax2.set_ylabel('Occam factor', color='C1')
ax1.set_xlabel('Degree of polynomial')

fig.savefig('{}/{}_test'.format(outdir, label))









class Analytical1DLikelihood(Likelihood):
    """
    A general class for 1D analytical functions. The model
    parameters are inferred from the arguments of function

    Parameters
    ----------
    x, y: array_like
        The data to analyse
    func:
        The python function to fit to the data. Note, this must take the
        dependent variable as its first argument. The other arguments
        will require a prior and will be sampled over (unless a fixed
        value is given).
    """

    def __init__(self, x, y, func):
        parameters = infer_parameters_from_function(func)
        Likelihood.__init__(self, dict.fromkeys(parameters))
        self.x = x
        self.y = y
        self.__func = func
        self.__function_keys = list(self.parameters.keys())

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={})'.format(self.x, self.y, self.func.__name__)

    @property
    def func(self):
        """ Make func read-only """
        return self.__func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self.__function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers will be converted to arrays internally """
        return self.__x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self.__x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers will be converted to arrays internally """
        return self.__y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self.__y = y

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters)





def infer_parameters_from_function(func):
    """ Infers the arguments of function (except the first arg which is
        assumed to be the dep. variable)
    """
    parameters = inspect.getargspec(func).args
    parameters.pop(0)
    return parameters



class PoissonLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func):
        """
        A general Poisson likelihood for a rate - the model parameters are
        inferred from the arguments of function, which provides a rate.

        Parameters
        ----------

        x: array_like
            A dependent variable at which the Poisson rates will be calculated
        y: array_like
            The data to analyse - this must be a set of non-negative integers,
            each being the number of events within some interval.
        func:
            The python function providing the rate of events per interval to
            fit to the data. The function must be defined with the first
            argument being a dependent parameter (although this does not have
            to be used by the function if not required). The subsequent
            arguments will require priors and will be sampled over (unless a
            fixed value is given).
        """

        Analytical1DLikelihood.__init__(self, x=x, y=y, func=func)

    def log_likelihood(self):
        rate = self.func(self.x, **self.model_parameters)
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Poisson rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))
        elif np.any(rate < 0.):
            raise ValueError(("Poisson rate function returns a negative",
                              " value!"))
        elif np.any(rate == 0.):
            return -np.inf
        else:
            return np.sum(-rate + self.y * np.log(rate) - gammaln(self.y + 1))

    def __repr__(self):
        return Analytical1DLikelihood.__repr__(self)

    @property
    def y(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y

    @y.setter
    def y(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        # check array is a non-negative integer array
        if y.dtype.kind not in 'ui' or np.any(y < 0):
            raise ValueError("Data must be non-negative integers")
        self.__y = y
