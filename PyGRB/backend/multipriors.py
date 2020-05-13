import numpy as np

from bilby.core.prior import PriorDict        as bilbyPriorDict
from bilby.core.prior import Uniform          as bilbyUniform
from bilby.core.prior import LogUniform       as bilbyLogUniform
from bilby.core.prior import Constraint       as bilbyConstraint

from PyGRB.backend.makepriors import MakePriors

class MultiPriors(MakePriors):
    '''
        Doc string goes here.
    '''

    def __init__(self,
                        priors_pulse_start, priors_pulse_end,
                        lens, channels,
                        ## just a separating line
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
                        priors_sigma_lo     = 1e-3,
                        priors_sigma_hi     = 1e3,
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_scale_min    = 1e0,  ## SCALING IS COUNTS / BIN
                        priors_scale_max    = 1e5,  ## SCALING IS COUNTS / BIN
                        **kwargs):

        keys = []
        for k in channels:
            super(MakePriors, self).__init__(lens = lens, channel = k, **kwargs)
            keys += self.keys
        self.keys = keys
        self.priors = bilbyPriorDict(
                conversion_function = self._make_constraints_multi(channels))

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
        self.priors_sigma_lo     = priors_sigma_lo
        self.priors_sigma_hi     = priors_sigma_hi
        self.priors_scale_min    = priors_scale_min
        self.priors_scale_max    = priors_scale_max
        self.populate_priors()

    def _make_constraints_multi(self, channels):
        n = self.max_pulse + 1
        l = self.residual_list
        c_keys = ['a', 'b', 'c', 'd']
        def constraint_function(parameters):
            # accessing pulses directly by index
            for c in c_keys:
                for i in range(2, n):
                    con_key = f'constraint_{i}_{c}'
                    st_key1 = f'start_{i-1}_{c}'
                    st_key2 = f'start_{i}_{c}'
                    parameters[con_key] = parameters[st_key2] - parameters[st_key1]
                # accessing residuals through list of residual positions
                for k in range(1, len(l)):
                    con_key = f'constraint_{l[k]}_{c}_res'
                    st_key1 = f'res_begin_{l[k-1]}_{c}'
                    st_key2 = f'res_begin_{l[k]}_{c}'
                    parameters[con_key] = parameters[st_key2] - parameters[st_key1]
                return parameters
        return constraint_function

if __name__ == '__main__':
    pass
