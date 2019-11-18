from DynamicBilby import BilbyObject
import numpy as np
import bilby
import scipy.stats as stats
import scipy.special as special

import matplotlib.pyplot    as plt

from matplotlib import rc

import corner

from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood
from bilby.core.likelihood  import GaussianLikelihood as bilbyGaussianLikelihood

from skellam_likelihood import SkellamLikelihood


rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class ResidualAnalysis(BilbyObject):
    """docstring for ResidualAnalysis."""

    def __init__(self, **kwargs):
        super(ResidualAnalysis, self).__init__(**kwargs)

    def get_residuals(self, priors, rate_function, channels):

        figure, axes = plt.subplots()
        deltat     = np.diff(self.GRB.bin_left)
        for i in channels:
            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.result.read_in_result(filename=open_result)
            MAP = dict()
            for j in range(1, self.num_pulses + 1):
                try:
                    key = 'constraint_' + str(j)
                    del priors[key]
                except:
                    pass
            for parameter in priors:
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP[parameter] = summary.median

            MAP['t_0'] = float(self.GRB.bin_left[0])
            counts_fit = rate_function(np.diff(self.GRB.bin_left), **MAP).astype('int')
            residuals = self.GRB.counts[:,i] - counts_fit
            # axes.plot(  self.GRB.bin_left, residuals + i * 2e3,
                        # color = self.colours[i])

            residual_priors             = bilbyPriorDict()
            residual_priors['sg_A']     = bilbyLogUniform(1e-1,1e6,latex_label='$A$')
            residual_priors['sg_t_0']   = bilbyUniform(2,7,latex_label='$t_0$')
            residual_priors['sg_tau']   = bilbyLogUniform(1e-3,1e3,latex_label='$\\tau$')
            residual_priors['sg_omega'] = bilbyLogUniform(1e-3,1e3,latex_label='$\\omega$')
            residual_priors['sg_phi']   = bilbyUniform(-np.pi,np.pi,latex_label='$\\phi$')
            ## if sigma gets too large then it will just fit a straight line
            ## easier to be wrong in sigma than fit 5 params
            residual_priors['sigma']    = 1
            # residual_priors['sigma']    = bilbyUniform(0, 1e1, 'sigma')

            likelihood   = bilbyGaussianLikelihood( self.GRB.bin_left,
                                                    residuals,
                                                    self.sine_gaussian)

            ## PASS IN THE COUNTS AND FIT DIRECTLY
            # likelihood = SkellamLikelihood(
                            # x   = self.GRB.bin_left,
                            # y_1 = np.rint(self.GRB.counts[:,i]).astype('uint'),
                            # y_2 = counts_fit,
                            # func= self.sine_gaussian    )
                            # func= residuals_bessel)

            result_label = self.fstring + '_res_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.run_sampler( likelihood = likelihood,
                                        priors     = residual_priors,
                                        sampler    = self.sampler,
                                        nlive      = 80,
                                        outdir     = self.outdir,
                                        label      = result_label,
                                        save       = True)

            plotname = self.outdir + '/' + result_label +'_res_corner.pdf'
            result.plot_corner(filename = plotname)
            MAP2 = dict()
            for parameter in residual_priors:
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP2[parameter] = summary.median
            try:
                del MAP2['sigma']
            except:
                pass
            res_fit  = (self.sine_gaussian(self.GRB.bin_left,
                                        **MAP2) / widths)
            axes.plot(  self.GRB.bin_left, res_fit + i * 2e3, 'k-')

        axes.set_xlim(  left = self.GRB.bin_left[0],
                        right = self.GRB.bin_left[-1])
        plotname = self.outdir + '/' + self.fstring +'_residuals.pdf'
        figure.savefig(plotname)



def shitty_function():
    def sine_gaussian(times, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
        return (sg_A * np.exp(- np.square((times - sg_t_0) / sg_tau)) *
                np.cos(sg_omega * times + sg_phi) )
    injection_parameters = bilbyPriorDict()
    # injection_parameters['background'] = 100 * 0.064
    # injection_parameters['start_1']    = 2
    # injection_parameters['scale_1']    = 3e3 * 0.064
    # injection_parameters['tau_1']      = 2
    # injection_parameters['xi_1']       = 3

    # injection_parameters['A']    = 2
    # injection_parameters['Omega']= 2
    # injection_parameters['s']    = 2
    # injection_parameters['t_t']  = 2
    # injection_parameters['Delta']= 2

    injection_parameters['sg_A']    = 5e3 * 0.064
    injection_parameters['sg_t_0']  = 4
    injection_parameters['sg_tau']  = 1
    injection_parameters['sg_omega']= 2
    injection_parameters['sg_phi']  = -1

    times = np.linspace(-2, 10, 1000)

    # res = residuals_bessel(times, **injection_parameters)
    res = sine_gaussian(times, **injection_parameters)

    fig, ax = plt.subplots(figsize = (6, 4), constrained_layout=True)
    ax.plot(times, res, linewidth = 0.5)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('counts / sec')
    plt.savefig('testtesttest.pdf')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    args = parser.parse_args()
    HPC = args.HPC

    if not HPC:
        rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        SAMPLER = 'Nestle'
    else:
        SAMPLER = 'dynesty'


    shitty_function()
    Trigger = ResidualAnalysis(trigger = 999, times = (3.5, 5),
                datatype = 'discsc', nSamples = 500, sampler = 'Nestle',
                priors_pulse_start = 3, priors_pulse_end = 7)

    Trigger.one_FRED(channels = [0,1,2,3], test = False, plot = False)
    # Trigger.num_pulses = 2
    # Trigger.tlabel     = Trigger.get_trigger_label()
    # Trigger.fstring    = Trigger.get_file_string()
    # Trigger.outdir     = Trigger.get_directory_name()
    Trigger.get_residuals(  priors = Trigger.priors.copy(),
                            rate_function = Trigger.one_FRED_rate,
                            channels = [0,1,2,3])


def residuals_bessel(times, A, Omega, s, t_t, Delta):
    return (np.where(times > t_t + Delta / 2,
            A * special.j0(s * Omega * (- t_t + times - Delta / 2) ),
           (np.where(times < t_t - Delta / 2,
            A * special.j0(    Omega * (  t_t - times - Delta / 2) ), A))) )

def sine_gaussian( times, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
    return (sg_A * np.exp(- np.square((times - sg_t_0) / sg_tau)) *
            np.cos(sg_omega * times + sg_phi) )









    print(injection_parameters)


#
# if __name__ == '__main__':
#     nSamples = 200
#     sampler  = 'nestle'
#     test = BilbyObject(trigger = 1, times = (-2, 50), test = True,
#                 datatype = 'discsc', nSamples = nSamples, sampler = sampler,
#                 priors_pulse_start = -5, priors_pulse_end = 50,
#                 priors_td_lo = 0,  priors_td_hi = 30)
#     test.inject_signal(residuals = 'Bessel')
