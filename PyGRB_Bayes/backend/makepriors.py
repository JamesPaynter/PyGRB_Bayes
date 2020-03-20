import numpy as np

from bilby.core.prior import PriorDict        as bilbyPriorDict
from bilby.core.prior import Uniform          as bilbyUniform
from bilby.core.prior import LogUniform       as bilbyLogUniform
from bilby.core.prior import Constraint       as bilbyConstraint

from PyGRB_Bayes.backend.makekeys import MakeKeys

class MakePriors(MakeKeys):
    '''
        Doc string goes here.
    '''

    def __init__(self,
                        priors_pulse_start, priors_pulse_end,
                        lens, channel,
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
                        priors_nu_min       = 1e-1,
                        priors_nu_max       = 1e1,
                        priors_sigma_lo     = 1e-3,
                        priors_sigma_hi     = 1e3,
                        priors_scale_min    = 1e0,  ## SCALING IS COUNTS / BIN
                        priors_scale_max    = 1e5,  ## SCALING IS COUNTS / BIN
                        **kwargs):

        super(MakePriors, self).__init__(lens = lens,channel = channel,**kwargs)
        self.priors = bilbyPriorDict(
                        conversion_function = self._make_constraints())
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

    def _make_constraints(self):
        n = self.max_pulse + 1
        l = self.residual_list
        def constraint_function(parameters):
            # accessing pulses directly by index
            for i in range(2, n):
                con_key = f'constraint_{i}_{self.c}'
                st_key1 = f'start_{i-1}_{self.c}'
                st_key2 = f'start_{i}_{self.c}'
                parameters[con_key] = parameters[st_key2] - parameters[st_key1]
            # accessing residuals through list of residual positions
            for k in range(1, len(l)):
                con_key = f'constraint_{l[k]}_{self.c}_res'
                st_key1 = f'res_begin_{l[k-1]}_{self.c}'
                st_key2 = f'res_begin_{l[k]}_{self.c}'
                parameters[con_key] = parameters[st_key2] - parameters[st_key1]
            return parameters
        return constraint_function

    def populate_priors(self):
        ''' initialise priors

            Pass in **kwargs, then overwrite pulse parameters as
            applicable. Otherwise take generic parameters defined in init.

            just make an overwrite prior function later

            add kwargs to list ??
        '''
        for key in self.keys:
            # find integer in key and put in label
            n = ''.join([char for char in key if char.isdigit()])
            self._make_prior(key, n, key[-1])

    def _make_prior(self, key: str, n: str, k: str):
        # where n is an integer given in string format.
        if key == f'background_{k}':
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_bg_lo,
                maximum=self.priors_bg_hi,
                latex_label = f'B {k}',
                unit='counts / sec')

        elif key == 'time_delay':
            self.priors[key] = bilbyUniform(
                minimum=self.priors_td_lo,
                maximum=self.priors_td_hi,
                latex_label='$\\Delta t$',
                unit=' seconds ')
            ## throw error if self.lens is False

        elif key == 'magnification_ratio':
            self.priors[key] = bilbyUniform(
                minimum=self.priors_mr_lo,
                maximum=self.priors_mr_hi,
                latex_label='$\\Delta \\mu$',
                unit=' ')
            ## throw error if self.lens is False

        elif 'start' in key:
            self.priors[key] = bilbyUniform(
                minimum=self.priors_pulse_start,
                maximum=self.priors_pulse_end,
                latex_label=f'$\\Delta_{n} {k}$', unit='sec')
            if int(n) > 1:
                c_key = f'constraint_{n}_{k}'
                self.priors[c_key] = bilbyConstraint(
                    minimum=0,
                    maximum=float(self.priors_pulse_end -
                                  self.priors_pulse_start))

        elif 'scale' in key:
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_scale_min,
                maximum=self.priors_scale_max,
                latex_label=f'$A_{n} {k}$', unit='counts / sec')

        elif 'tau' in key:
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_tau_lo,
                maximum=self.priors_tau_hi,
                latex_label = f'$\\tau_{n} {k}$', unit=' ')

        elif 'xi' in key:
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_xi_lo,
                maximum=self.priors_xi_hi,
                latex_label=f'$\\xi_{n} {k}$', unit=' ')

        elif 'gamma' in key:
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_gamma_min,
                maximum=self.priors_gamma_max,
                latex_label=f'$\\gamma_{n} {k}$', unit=' ')

        elif 'nu' in key:
            self.priors[key] = bilbyLogUniform(
                minimum=self.priors_nu_min,
                maximum=self.priors_nu_max,
                latex_label=f'$\\nu_{n} {k}$', unit=' ')

        elif 'sigma' in key:
            self.priors[key] = bilbyLogUniform(
                minimum = self.priors_sigma_lo,
                maximum = self.priors_sigma_hi,
                latex_label= f'$\\sigma_{n} {k}$', unit = ' ')

        elif 'begin' in key:
            self.priors[key] = bilbyUniform(
                minimum=self.priors_pulse_start,
                maximum=self.priors_pulse_end,
                latex_label=f'$\\delta_{n} {k}$', unit='sec')
            if int(n) > 1:
                c_key = f'constraint_{n}_{k}_res'
                self.priors[c_key] = bilbyConstraint(
                    minimum=0,
                    maximum=float(self.priors_pulse_end -
                                  self.priors_pulse_start))

        elif 'sg_A' in key:
            self.priors[key] = bilbyLogUniform(1e0, 1e3,
                                latex_label=f'res $A_{n} {k}$')
        elif 'sg_lambda' in key:
            self.priors[key] = bilbyLogUniform(1e-3, 1e3,
                                latex_label = f'res $\\lambda_{n} {k}$')
        elif 'sg_omega' in key:
            self.priors[key] = bilbyLogUniform(1e-3, 1e4,
                                latex_label = f'res $\\omega_{n} {k}$')
        elif 'sg_phi' in key:
            self.priors[key] = bilbyUniform(-np.pi, np.pi,
                                latex_label = f'res $\\phi_{n} {k}$')
        elif 'bes_A' in key:
            self.priors[key] = bilbyLogUniform(1e-1, 1e6,
                                latex_label = f'res $A_{n} {k}$')
        elif 'bes_Omega' in key:
            self.priors[key] = bilbyLogUniform(1e-3, 1e3,
                                latex_label = f'res $\\Omega_{n} {k}$')
        elif 'bes_s' in key:
            self.priors[key] = bilbyLogUniform(1e-3, 1e3,
                                latex_label = f'res $s_{n} {k}$')
        elif 'bes_Delta' in key:
            self.priors[key] = bilbyUniform(-np.pi, np.pi,
                                latex_label = f'res $\\Delta_{n} {k}$')
        else:
            raise Exception(f'Key not found : {key}')

    def return_prior_dict(self):
        return self.priors

if __name__ == '__main__':
    pass
