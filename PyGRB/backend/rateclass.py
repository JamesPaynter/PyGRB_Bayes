import numpy as np
from scipy.special import gammaln

from bilby import Likelihood as bilbyLikelihood

from PyGRB.backend.makekeys import MakeKeys
from PyGRB.backend.rate_functions import *


class PoissonRate(MakeKeys, bilbyLikelihood):
    def __init__(self, x, y, channel, lens, **kwargs):

        '''
            Doc string goes here.
            kwargs is there because sometime model dict
            comes with a name.
        '''
        super(PoissonRate, self).__init__(  lens = lens, channel = channel,
                                            **kwargs)
        self.x = x
        self.y = y
        self.parameters = {k: None for k in self.keys} ## creates a dict
        self.rate_lists = [gaussian_pulse, FRED_pulse, FREDx_pulse,
                           convolution_gaussian, sine_gaussian, modified_bessel]

    @staticmethod
    def calculate_rate(x, parameters, pulse_arr, key_list, rate_function, k):
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
                p_key       = f'{key}_{j}_{k}'
                kwargs[key] = parameters[p_key]
            rates += rate_function(**kwargs)
        return rates

    @staticmethod
    def calculate_rate_lens(x, parameters, pulse_arr, key_list, rate_function, k):
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs   = { 'times' : x}
            l_kwargs = { 'times' : x}
            for key in key_list:
                p_key           = f'{key}_{j}_{k}'
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
        for count_list, p_list, rate in zip(
        self.rate_counts, self.param_lists, self.rate_lists):
            rates+= return_rate(x, parameters, count_list, p_list, rate, self.c)
        try:
            rates += parameters[f'background_{self.c}']
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

    def return_line_from_sample(self, sample_dict):
        if self.lens:
            rate = self._sum_rates(self.x, sample_dict, self.calculate_rate_lens)
        else:
            rate = self._sum_rates(self.x, sample_dict, self.calculate_rate)
        return rate

if __name__ == '__main__':
    pass
