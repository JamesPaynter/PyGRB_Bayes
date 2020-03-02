import numpy as np
from scipy.special import gammaln

from bilby import Likelihood as bilbyLikelihood

from PyGRB_Bayes.backend.makekeys import MakeKeys
from PyGRB_Bayes.backend.rate_functions import *


class PoissonRate(MakeKeys, bilbyLikelihood):
    def __init__(self, x, y,    count_FRED, count_FREDx,
                                count_sg, count_bes,
                                lens, **kwargs):

        '''
            Doc string goes here.
            kwargs is there because sometime model dict
            comes with a name.
        '''
        super(PoissonRate, self).__init__(  count_FRED   = count_FRED,
                                            count_FREDx  = count_FREDx,
                                            count_sg  = count_sg,
                                            count_bes = count_bes,
                                            lens = lens)
        self.x = x
        self.y = y
        self.parameters = {k: None for k in self.keys} ## creates a dict

    @staticmethod
    def calculate_rate(x, parameters, pulse_arr, key_list, rate_function):
        ''' finished by putting in lens func below

            x : series of points for function to be evaluated at.

            parameters : dictionary of parameters from the sampler to be passed
                         into the rate function.

            pulse_arr : the array (list) of pulse keys (eg. [1, 3, 5]). These
                        are then appened to the keys in key_list.

            key_list  : the list of generic keys appropriate for the rate
                        function.

            rate_function : the pulse / residual function through which all the
                            parameters are passed.
        '''
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs = { 'times' : x}
            for key in key_list:
                p_key       = f'{key}_{j}'
                kwargs[key] = parameters[p_key]
            rates += rate_function(**kwargs)
        return rates

    @staticmethod
    def calculate_rate_lens(x, parameters, pulse_arr, key_list, rate_function):
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs   = { 'times' : x}
            l_kwargs = { 'times' : x}
            for key in key_list:
                p_key           = f'{key}_{j}'
                kwargs[key]     = parameters[p_key]
                l_kwargs[key]   = parameters[p_key]
            rates += rate_function(**kwargs)
            try:
                l_kwargs['start'] = l_kwargs['start'] + parameters['time_delay']
            except:
                pass
            try:
                l_kwargs['res_begin'] = l_kwargs['res_begin'] + parameters['time_delay']
            except:
                pass
            rates += rate_function(**l_kwargs) * parameters['magnification_ratio']
        return rates

    def _sum_rates(self, x, parameters, return_rate):
        rates = np.zeros(len(x))
        rates+= return_rate(x, parameters,     self.count_FRED,
                            self.FRED_list,    FRED_pulse)
        rates+= return_rate(x, parameters,     self.count_FREDx,
                            self.FREDx_list,   FREDx_pulse)
        rates+= return_rate(x, parameters,     self.count_sg,
                            self.res_sg_list,  sine_gaussian)
        rates+= return_rate(x, parameters,     self.count_bes,
                            self.res_bes_list, modified_bessel)
        # for count_list, p_list, rate in zip(
        # self.count_lists, self.param_lists, self.rate_lists):
        #     rates+= return_rate(x, parameters, count_list, p_list, rate)
        try:
            rates += parameters['background']
        except:
            pass
        return np.where(np.any(rates < 0.), 0, rates)


    def log_likelihood(self):
        if self.lens:
            rate = self._sum_rates(self.x, self.parameters, self.calculate_rate_lens)
        else:
            rate = self._sum_rates(self.x, self.parameters, self.calculate_rate)

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


if __name__ == '__main__':
    pass
