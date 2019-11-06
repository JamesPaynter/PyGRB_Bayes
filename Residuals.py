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

            residual_priors             = bilbyPriorDict()
            residual_priors['sg_A']     = bilbyLogUniform(1e-2,1e2,latex_label='$A$')
            residual_priors['sg_t_0']   = bilbyUniform(0,10,latex_label='$t_0$')
            residual_priors['sg_tau']   = bilbyLogUniform(1e-2,1e2,latex_label='$\\tau$')
            residual_priors['sg_omega'] = bilbyLogUniform(1e-1,1e2,latex_label='$\\omega$')
            residual_priors['sg_phi']   = bilbyUniform(-np.pi,np.pi,latex_label='$\\phi$')
            ## if sigma gets too large then it will just fit a straight line
            ## easier to be wrong in sigma than fit 5 params
            residual_priors['sigma']    = bilbyUniform(0, 1e2, 'sigma')

            # residual_counts = np.rint(residuals * 0.064).astype('uint')

            likelihood   = bilbyGaussianLikelihood( self.GRB.bin_left, residuals,
                                                    self.sine_gaussian)

            result_label = self.fstring + '_res_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.run_sampler( likelihood = likelihood,
                                        priors     = residual_priors,
                                        sampler    = self.sampler,
                                        nlive      = 200,
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


    @staticmethod
    def sine_gaussian( time, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
        return (sg_A * np.exp(- np.square((time - sg_t_0) / sg_tau)) *
                np.cos(sg_omega * time + sg_phi) )


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
