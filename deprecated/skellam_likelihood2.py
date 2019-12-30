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
        print('here')
        if not isinstance(rate, np.ndarray):
            raise ValueError(
                "Skellam rate function returns wrong value type! "
                "Is {} when it should be numpy.ndarray".format(type(rate)))

                # is this correct P(k) = 0 for lambda = 0? (Poisson)
        # return  np.where(self.y_1 == 0 and self.y_2 == 0, 0,
        print('  x y1 y2 rate/2 lny1/y2 sqrt logiv')
        for i in range(10):
            print('%.2f %i %i %.2f %.2f %.2f %.2f %.2f' %
            (self.x[i], self.y_1[i], self.y_2[i],
            rate[i] / 2,
            np.log(self.y_1[i] / self.y_2[i]),
            np.sqrt(self.y_1[i] * self.y_2[i]),
            np.log(special.iv(self.x[i], 2*np.sqrt(self.y_1[i] * self.y_2[i]))),
            ( - self.y_1[i] - self.y_2[i]
                            + rate[i] / 2 * np.log(self.y_1[i] / self.y_2[i])
                            + np.log(special.iv(self.x[i], 2
                            * np.sqrt(self.y_1[i] * self.y_2[i]))))
            ))
        # return  np.where(np.all(self.y_2 == 0, self.y_1 == 0), 0,
        return  np.where(self.y_2 == 0,
                ## poisson rate for y_2
                np.sum(-rate + self.y_1 * np.log(rate) - gammaln(self.y_1 + 1)),
                np.where(self.y_1 == 0,
                ## poisson rate for y_1
                np.sum(-rate + self.y_2 * np.log(rate) - gammaln(self.y_2 + 1)),
                ## skellam rate
                np.sum(  - self.y_1 - self.y_2
                                + rate / 2 * np.log(self.y_1 / self.y_2)
                                + np.log(special.iv(self.x, 2
                                * np.sqrt(self.y_1 * self.y_2))))
                ))
        # return  np.where(y_1 == 0 and y_2 == 0, 'zero',
        # return    np.where(self.y_1 == 0, 3,
        #         np.where(self.y_2 == 0, 2,
        #         1
        #         ))
        #
        #
        # LL =  np.sum(  - self.y_1 - self.y_2
        #                 + rate / 2 * np.log(self.y_1 / self.y_2)
        #                 + np.log(special.iv(self.x, 2
        #                 * np.sqrt(self.y_1 * self.y_2)))
        #              )
        # print('\n******\n')
        # print(LL)
        # print('******\n')
        #
        # return LL
        #
        # return


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
