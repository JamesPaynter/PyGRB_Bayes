from DynamicBilby import BilbyObject
import numpy as np
import bilby
import scipy.stats as stats

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
            widths = self.GRB.bin_right - self.GRB.bin_left
            rates_fit = rate_function(np.diff(self.GRB.bin_left), **MAP) / widths
            residuals = self.GRB.rates[:,i] - rates_fit
            axes.plot(  self.GRB.bin_left, residuals + i * 2e3,
                        color = self.colours[i])

            self.res_sine_gaussian_priors()
            self.residual_priors['t_0'] = bilbyDeltaFunction(
                                peak = float(self.GRB.bin_left[0]), name = None,
                                latex_label = None, unit = None )
            residual_counts = np.rint(residuals * 0.064).astype('uint')
            likelihood   = bilbyGaussianLikelihood(  deltat, residual_counts,
                                                    self.sine_gaussian)
            result_label = self.fstring + '_res_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.run_sampler( likelihood = likelihood,
                                        priors     = self.residual_priors,
                                        sampler    = self.sampler,
                                        nlive      = self.nSamples,
                                        outdir     = self.outdir,
                                        label      = result_label,
                                        save       = True)
            try:
                del self.residual_priors['t_0']
            except:
                pass

            plotname = self.outdir + '/' + result_label +'_res_corner.pdf'
            result.plot_corner(filename = plotname)
            priors = self.residual_priors.copy()
            MAP2 = dict()
            for j in range(1, self.num_pulses + 1):
                try:
                    key = 'constraint_' + str(j)
                    del priors[key]
                except:
                    pass
            for parameter in priors:
                summary = result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP2[parameter] = summary.median
            res_fit  = (self.sine_gaussian(np.diff(self.GRB.bin_left),
                                        **MAP2) / widths)



            axes.plot(  self.GRB.bin_left, res_fit + i * 2e3, 'k-')

        axes.set_xlim(  left = self.GRB.bin_left[0],
                        right = self.GRB.bin_left[-1])

        figure.savefig('residuals.pdf')


    @staticmethod
    def sine_gaussian( dt, t_0, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
        times = np.cumsum(dt)
        times = np.insert(times, 0, 0.0)
        times+= t_0
        return (sg_A * np.exp(- np.square((times - sg_t_0) / sg_tau)) *
                np.cos(sg_omega + sg_phi) )

    def res_sine_gaussian_priors(self):
        self.residual_priors = bilbyPriorDict()
        self.residual_priors['sg_A']     = bilbyLogUniform(1e-2,1e2,latex_label='$A$')
        self.residual_priors['sg_t_0']   = bilbyUniform(3,7,latex_label='$t_0$')
        self.residual_priors['sg_tau']   = bilbyLogUniform(1e-2,1e2,latex_label='$\\tau$')
        self.residual_priors['sg_omega'] = bilbyLogUniform(1e-2,1e2,latex_label='$\\omega$')
        self.residual_priors['sigma'] = bilbyLogUniform(1e-2,1e2,latex_label='$\\sigma$')
        self.residual_priors['sg_phi']   = bilbyUniform(0,2*np.pi,latex_label='$\\phi$')


if __name__ == '__main__':
    Trigger = ResidualAnalysis(trigger = 999, times = (3, 7),
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
