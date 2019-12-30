import bilby
import inspect
import numpy as np
import scipy.special as special
from scipy.special import gammaln



class SkellamLikelihood(bilby.Likelihood):
    def __init__(self, x, y_1, y_2, func):
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

        super(SkellamLikelihood, self).__init__()
        self.x   = x
        self.y_1 = y_1
        self.y_2 = y_2
        self.func= func

         # These lines of code infer the parameters from the provided function
        parameters = inspect.getfullargspec(func).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)
        self.function_keys = self.parameters.keys()

    def log_likelihood(self):
        rate = self.func(self.x, **self.parameters)
        model_parameters = {k: self.parameters[k] for k in self.function_keys}
        # res = self.y - self.func(self.x, **model_parameters)
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Skellam rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))

        def inner_function(x, y_1, y_2, func, rates):
            if y_2 == 0:
                return np.sum(-rate + y_1 * np.log(rate) - gammaln(y_1 + 1))
            elif y_1 == 0:
                return np.sum(-rate + y_2 * np.log(rate) - gammaln(y_2 + 1))
            else:
                return np.sum(  - y_1 - y_2
                                + rates / 2 * np.log(y_1 / y_2)
                                + np.log(special.iv(x, 2
                                * np.sqrt(y_1 * y_2))))
        vfunc = np.vectorize(inner_function)
        return np.sum(vfunc(self.x, self.y_1, self.y_2, self.func, rate))

    # def __repr__(self):
    #     return 'SkellamLikelihood(x={}, y_1={}, y_2={}, func={})'.format(
    #     self.x, self.y_1, self.y_2, self.func.__name__)

    @property
    def y_1(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y_1

    @y_1.setter
    def y_1(self, y_1):
        if not isinstance(y_1, np.ndarray):
            y_1 = np.array([y_1])
        # check array is a non-negative integer array
        if y_1.dtype.kind not in 'ui' or np.any(y_1 < 0):
            raise ValueError("Data must be non-negative integers")
        self.__y_1 = y_1

    @property
    def y_2(self):
        """ Property assures that y-value is a positive integer. """
        return self.__y_2

    @y_2.setter
    def y_2(self, y_2):
        if not isinstance(y_2, np.ndarray):
            y_2 = np.array([y_2])
        # check array is a non-negative integer array
        if y_2.dtype.kind not in 'ui' or np.any(y_2 < 0):
            raise ValueError("Data must be non-negative integers")
        self.__y_2 = y_2




if __name__ == '__main__':

    dt = np.ones(100)
