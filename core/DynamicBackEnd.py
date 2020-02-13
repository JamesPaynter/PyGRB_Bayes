import sys
import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import scipy.special as special
from scipy.special import gammaln

import bilby
from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import Analytical1DLikelihood
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood

MIN_FLOAT = sys.float_info[3]

class MakeKeys(object):
    '''
        Doc string goes here.
    '''

    def __init__(self,  count_FRED = [], count_FREDx = [],
                        count_sg = [], count_bes = [],
                        lens = False, **kwargs):
        super(MakeKeys, self).__init__()
        self.count_FRED   = count_FRED
        self.count_FREDx  = count_FREDx
        self.count_sg     = count_sg
        self.count_bes    = count_bes

        self.FRED_list    = ['start', 'scale', 'tau', 'xi']
        self.FREDx_list   = self.FRED_list.copy() + ['gamma', 'nu']
        self.res_sg_list  = ['sg_A', 'res_begin', 'sg_lambda', 'sg_omega', 'sg_phi']
        self.res_bes_list = [   'bes_A', 'bes_Omega', 'bes_s',
                                'res_begin', 'bes_Delta']

        self.lens         = lens
        self.lens_list    = ['time_delay', 'magnification_ratio']

        self.keys = []
        self.get_max_pulse()
        self.get_residual_list()
        self.fill_keys_list()

    def fill_list(self, list, array):
        return ['{}_{}'.format(list[k], i) for k in range(len(list))
                                           for i in array]

    def fill_keys_list(self):
        if self.lens:
            self.keys += self.lens_list
        self.keys += ['background']
        self.keys += self.fill_list(self.FRED_list,  self.count_FRED)
        self.keys += self.fill_list(self.FREDx_list, self.count_FREDx)
        self.keys += self.fill_list(self.res_sg_list,  self.count_sg)
        self.keys += self.fill_list(self.res_bes_list, self.count_bes)

    def get_max_pulse(self):
        mylist = self.count_FRED + self.count_FREDx
        ## set gets the unique values of the list
        myset  = set(mylist)
        try:
            self.max_pulse = max(myset) ## WILL NEED EXPANDING
        except:
            self.max_pulse = 0

    def get_residual_list(self):
        mylist = self.count_sg + self.count_bes
        myarr  = np.array(mylist)
        myarr  = np.unique(myarr)
        mysort = np.sort(myarr)
        mylist = [mysort[i] for i in range(len(mysort))]
        self.residual_list = mylist

# @dataclass
# class PriorRanges:
#     priors_pulse_start: float
#     priors_pulse_end:   float
#     priors_td_lo:       float = None
#     priors_td_hi:       float = None
#     priors_bg_lo:       float = 1e-1  ## SCALING IS COUNTS / BIN
#     priors_bg_hi:       float = 1e3   ## SCALING IS COUNTS / BIN
#     priors_mr_lo:       float = 0.2   ## which means that it is
#     priors_mr_hi:       float = 1.4     # 1 / 0.064 times smaller
#     priors_tau_lo:      float = 1e-3  # than you think it is
#     priors_tau_hi:      float = 1e3   # going to be !!!!!!!!!!!!
#     priors_xi_lo:       float = 1e-3
#     priors_xi_hi:       float = 1e3
#     priors_gamma_min:   float = 1e-1
#     priors_gamma_max:   float = 1e1
#     priors_nu_min:      float = 1e-1
#     priors_nu_max:      float = 1e1
#     priors_scale_min:   float = 1e0  ## SCALING IS COUNTS / BIN
#     priors_scale_max:   float = 1e4


class MakePriors(MakeKeys):
    '''
        Doc string goes here.
    '''

    def __init__(self,
                        ## just a separating line
                        priors_pulse_start, priors_pulse_end,
                        # count_FRED, count_sg, lens, ## now in **kwargs
                        priors_td_lo = None,
                        priors_td_hi = None,
                        priors_bg_lo        = 1e-1,  ## SCALING IS COUNTS / BIN
                        priors_bg_hi        = 1e3,   ## SCALING IS COUNTS / BIN
                        priors_mr_lo        = 0.2,   ## which means that it is
                        priors_mr_hi        = 1.4,   # 1 / 0.064 times smaller
                        priors_tau_lo       = 1e-3,  # than you think it is
                        priors_tau_hi       = 1e3,   # going to be !!!!!!!!!!!!
                        priors_xi_lo        = 1e-3,
                        priors_xi_hi        = 1e3,
                        priors_gamma_min    = 1e-1,
                        priors_gamma_max    = 1e1,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e0,  ## SCALING IS COUNTS / BIN
                        priors_scale_max    = 1e5,
                        **kwargs):  ## SCALING IS COUNTS / BIN):
        # super(MakePriors, self).__init__(count_FRED, count_sg, lens)
        super(MakePriors, self).__init__(**kwargs)

        self.priors = bilbyPriorDict(conversion_function = self.make_constraints())

        self.priors_pulse_start  = priors_pulse_start
        self.priors_pulse_end    = priors_pulse_end
        self.priors_bg_lo        = priors_bg_lo
        self.priors_bg_hi        = priors_bg_hi
        self.priors_td_lo        = priors_td_lo
        self.priors_td_hi        = priors_td_hi
        self.priors_mr_lo        = priors_mr_lo
        self.priors_mr_hi        = priors_mr_hi
        self.priors_tau_lo       = priors_tau_lo
        self.priors_tau_hi       = priors_tau_hi
        self.priors_xi_lo        = priors_xi_lo
        self.priors_xi_hi        = priors_xi_hi
        self.priors_gamma_min    = priors_gamma_min
        self.priors_gamma_max    = priors_gamma_max
        self.priors_nu_min       = priors_nu_min
        self.priors_nu_max       = priors_nu_max
        self.priors_scale_min    = priors_scale_min
        self.priors_scale_max    = priors_scale_max
        self.populate_priors()

    def make_constraints(self):
        n = self.max_pulse + 1
        l = self.residual_list
        def constraint_function(parameters):
            for i in range(2, n):
                con_key = f'constraint_{i}'
                st_key1 = f'start_{i-1}'
                st_key2 = f'start_{i}'
                parameters[con_key] = parameters[st_key2] - parameters[st_key1]
            for k in range(1, len(l)):
                con_key = f'constraint_{l[k]}_res'
                st_key1 = f'res_begin_{l[k-1]}'
                st_key2 = f'res_begin_{l[k]}'
                parameters[con_key] = parameters[st_key2] - parameters[st_key1]
            return parameters
        return constraint_function

    def populate_priors(self):
        ''' initialise priors

            Pass in **kwargs, then overwrite pulse parameters as
            applicable. Otherwise take generic parameters defined in init.

            add kwargs to list ??
        '''
        for key in self.keys:
            ## only works up to 9 pulses....
            for i in range(1, self.max_pulse + 1):
                if str(i) in key:
                    n = str(i)
                else:
                    pass
            for i in self.residual_list:
                if str(i) in key:
                    n = str(i)
                else:
                    pass
            ## find integer in key and put in label
            if key == 'background':
                self.priors[key] = bilbyLogUniform(
                minimum = self.priors_bg_lo,
                maximum = self.priors_bg_hi,
                latex_label='B',
                unit = 'counts / sec')

            elif key == 'time_delay':
                self.priors[key] = bilbyUniform(
                minimum = self.priors_td_lo,
                maximum = self.priors_td_hi,
                latex_label='$\\Delta t$',
                unit = ' seconds ')
                ## throw error if self.lens is False

            elif key == 'magnification_ratio':
                self.priors[key] = bilbyUniform(
                minimum = self.priors_mr_lo,
                maximum = self.priors_mr_hi,
                latex_label='$\\Delta \\mu$',
                unit = ' ')
                ## throw error if self.lens is False

            elif 'start' in key:
                self.priors[key] = bilbyUniform(
                    minimum = self.priors_pulse_start,
                    maximum = self.priors_pulse_end,
                    latex_label = '$\\Delta_{}$'.format(n), unit = 'sec')
                if int(n) > 1:
                    c_key = 'constraint_{}'.format(n)
                    self.priors[c_key] = bilbyConstraint(
                        minimum = 0,
                        maximum = float(self.priors_pulse_end -
                                        self.priors_pulse_start))

            elif 'scale' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_scale_min,
                    maximum = self.priors_scale_max,
                    latex_label='$A_{}$'.format(n), unit = 'counts / sec')

            elif 'tau' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_tau_lo,
                    maximum = self.priors_tau_hi,
                    latex_label='$\\tau_{}$'.format(n), unit = ' ')

            elif 'xi' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_xi_lo,
                    maximum = self.priors_xi_hi,
                    latex_label='$\\xi_{}$'.format(n), unit = ' ')

            elif 'gamma' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_gamma_min,
                    maximum = self.priors_gamma_max,
                    latex_label='$\\gamma_{}$'.format(n), unit = ' ')

            elif 'nu' in key:
                self.priors[key] = bilbyLogUniform(
                    minimum = self.priors_nu_min,
                    maximum = self.priors_nu_max,
                    latex_label='$\\nu_{}$'.format(n), unit = ' ')

            # elif 'sigma' in key:
                # print('Sigma priors not set')
                # self.priors[key] = bilbyLogUniform(
                #     minimum = self.priors_xi_lo,
                #     maximum = self.priors_xi_hi,
                #     latex_label= '$\\sigma_{}'.format(n), unit = ' ')

            elif 'begin' in key:
                self.priors[key] = bilbyUniform(
                    minimum = self.priors_pulse_start,
                    maximum = self.priors_pulse_end,
                    latex_label = '$\\delta_{}$'.format(n), unit = 'sec')
                if int(n) > 1:
                    c_key = 'constraint_{}_res'.format(n)
                    self.priors[c_key] = bilbyConstraint(
                        minimum = 0,
                        maximum = float(self.priors_pulse_end -
                                        self.priors_pulse_start) )

            elif 'sg_A' in key:
                self.priors[key] = bilbyLogUniform(1e1,1e3,latex_label='res $A$')

            elif 'sg_lambda' in key:
                self.priors[key] = bilbyLogUniform(1e-2,1e2,latex_label='res $\\lambda$')

            elif 'sg_omega' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e2,latex_label='res $\\omega$')

            elif 'sg_phi' in key:
                self.priors[key] = bilbyUniform(-np.pi,np.pi,latex_label='res $\\phi$')

            elif 'bes_A' in key:
                self.priors[key] = bilbyLogUniform(1e-1,1e6,latex_label='res $A$')

            elif 'bes_Omega' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $\\Omega$')

            elif 'bes_s' in key:
                self.priors[key] = bilbyLogUniform(1e-3,1e3,latex_label='res $s$')

            elif 'bes_Delta' in key:
                self.priors[key] = bilbyUniform(-np.pi,np.pi,latex_label='res $\\Delta$')

            else:
                raise Exception('Key not found : {}'.format(key))

    def return_prior_dict(self):
        return self.priors


class PoissonRate(MakeKeys, bilby.Likelihood):
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
    def FRED_pulse(times, start, scale, tau, xi):
        return np.where(times - start <= 0, MIN_FLOAT, scale * np.exp(
        - xi * ( (tau / (times - start)) + ((times - start) / tau) - 2)))

    @staticmethod
    def FREDx_pulse(times, start, scale, tau, xi, gamma, nu):
        return np.where(times - start <= 0, MIN_FLOAT, scale * np.exp(
        - np.power(xi * (tau / (times - start)), gamma)
        - np.power(xi * ((times - start) / tau), nu) - 2) )

    @staticmethod
    def sine_gaussian(times, sg_A, res_begin, sg_lambda, sg_omega, sg_phi):
        return (sg_A * np.exp(- np.square((times - res_begin) / sg_lambda)) *
                np.cos(sg_omega * times + sg_phi) )

    @staticmethod
    def count_bessel(times, bes_A, bes_Omega, bes_s, res_begin, bes_Delta):
        return np.where(times > res_begin + bes_Delta / 2,
                bes_A * special.j0(bes_s * bes_Omega *
               (- res_begin + times - bes_Delta / 2) ),
               (np.where(times < res_begin - bes_Delta / 2,
                bes_A * special.j0(bes_Omega *
               (res_begin - times - bes_Delta / 2) ),
               bes_A)))

    @staticmethod
    def insert_name(x, parameters, pulse_arr, key_list, rate_function):
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
                p_key      = key + f'_{j}'
                kwargs[key] = parameters[p_key]
            rates += rate_function(**kwargs)
        return rates

    @staticmethod
    def insert_name_lens(x, parameters, pulse_arr, key_list, rate_function):
        rates = np.zeros(len(x))
        for j in pulse_arr:
            kwargs   = { 'times' : x}
            l_kwargs = { 'times' : x}
            for key in key_list:
                p_key           = key + f'_{j}'
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

    def calculate_rate(self, x, parameters, insert_name_func):
        rates = np.zeros(len(x))
        rates+= insert_name_func(   x, parameters,     self.count_FRED,
                                    self.FRED_list,    self.FRED_pulse)
        rates+= insert_name_func(   x, parameters,     self.count_FREDx,
                                    self.FREDx_list,   self.FREDx_pulse)
        rates+= insert_name_func(   x, parameters,     self.count_sg,
                                    self.res_sg_list,  self.sine_gaussian)
        rates+= insert_name_func(   x, parameters,     self.count_bes,
                                    self.res_bes_list, self.count_bessel)
        try:
            rates += parameters['background']
        except:
            pass
        return np.where(np.any(rates < 0.), 0, rates)


    def log_likelihood(self):
        if self.lens:
            rate = self.calculate_rate(self.x, self.parameters, self.insert_name_lens)
        else:
            rate = self.calculate_rate(self.x, self.parameters, self.insert_name)

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
