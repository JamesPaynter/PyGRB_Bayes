from DynamicBilby import BilbyObject
import numpy as np
import scipy.stats as stats
import scipy.special as special

from matplotlib import rc
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec


import corner

import bilby
from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction
from bilby.core.likelihood  import PoissonLikelihood as bilbyPoissonLikelihood
from bilby.core.likelihood  import GaussianLikelihood as bilbyGaussianLikelihood

#from skellam_likelihood import SkellamLikelihood



class ResidualAnalysis(BilbyObject):
    """docstring for ResidualAnalysis."""

    def __init__(self, **kwargs):
        super(ResidualAnalysis, self).__init__(**kwargs)

    def get_residuals(self, priors, rate_function, channels):
        self.num_pulses = 1
        self.model  = 'one FRED'
        self.tlabel     = self.get_trigger_label()
        self.fstring    = self.get_file_string()
        self.outdir     = self.get_directory_name()

        deltat          = np.diff(self.GRB.bin_left)
        count_fits      = np.zeros((len(self.GRB.bin_left),4))
        residuals       = np.zeros((len(self.GRB.bin_left),4))
        residual_fits   = np.zeros((len(self.GRB.bin_left),4))
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
            count_fits[:,i] = counts_fit
            residuals[:,i] = self.GRB.counts[:,i] - counts_fit

            residual_priors             = bilbyPriorDict()
            residual_priors['sg_A']     = bilbyLogUniform(1e-1,1e6,latex_label='$A$')
            residual_priors['sg_t_0']   = bilbyUniform(2,7,latex_label='$t_0$')
            # residual_priors['sg_begin']   = bilbyUniform(2,7,latex_label='$t_0$')
            residual_priors['sg_tau']   = bilbyLogUniform(1e-3,1e3,latex_label='$\\tau$')
            residual_priors['sg_omega'] = bilbyLogUniform(1e-3,1e3,latex_label='$\\omega$')
            residual_priors['sg_phi']   = bilbyUniform(-np.pi,np.pi,latex_label='$\\phi$')
            ## if sigma gets too large then it will just fit a straight line
            ## easier to be wrong in sigma than fit 5 params
            residual_priors['sigma']    = 1

            likelihood   = bilbyGaussianLikelihood( self.GRB.bin_left,
                                                    residuals[:,i],
                                                    self.sine_gaussian)

            ## PASS IN THE COUNTS AND FIT DIRECTLY
            # likelihood = SkellamLikelihood(
                            # x   = self.GRB.bin_left,
                            # y_1 = np.rint(self.GRB.counts[:,i]).astype('uint'),
                            # y_2 = counts_fit,
                            # func= self.sine_gaussian    )
                            # func= residuals_bessel)

            res_result_label = self.fstring + '_res_result_' + self.clabels[i]
            res_open_result  = self.outdir + '/' + res_result_label +'_result.json'
            try:
                res_result = bilby.result.read_in_result(filename=res_open_result)
                print('Read in previous residual result.')
            except:
                print('Have to do the residuals this time.')
                res_result = bilby.run_sampler( likelihood = likelihood,
                                                priors     = residual_priors,
                                                sampler    = self.sampler,
                                                nlive      = 300,
                                                outdir     = self.outdir,
                                                label      = res_result_label,
                                                save       = True)


            plotname = self.outdir + '/' + res_result_label +'_res_corner.pdf'
            res_result.plot_corner(filename = plotname)
            MAP2 = dict()
            for parameter in residual_priors:
                summary = res_result.get_one_dimensional_median_and_error_bar(
                                parameter)
                MAP2[parameter] = summary.median
            try:
                del MAP2['sigma']
            except:
                pass
            res_fit  = self.sine_gaussian(self.GRB.bin_left, **MAP2)
            residual_fits[:,i] = res_fit

        widths = self.GRB.bin_right - self.GRB.bin_left
        rates  = self.GRB.counts        / widths[:,None]
        rates_fit       = count_fits    / widths[:,None]
        residual_rates  = residuals     / widths[:,None]
        res_fit_rates   = residual_fits / widths[:,None]

        self.new_plot(  x = self.GRB.bin_left, y = rates, y_fit = rates_fit,
                        channels = channels, y_res_fit = res_fit_rates)

        self.new_plot(  x = self.GRB.bin_left, y = residual_rates,
                        y_fit = res_fit_rates, channels = channels,
                        y_res_fit = None, residuals = True)

    def new_plot(   self, x, y, y_fit, channels,
                    y_res_fit = None, residuals = False, offsets = None):

        n_axes  = len(channels) + 1
        # n_axes  = min(np.shape(y)) + 1
        width   = 3.321
        height  = (width / 1.8) * 2
        heights = [5] + ([1 for i in range(n_axes - 1)])
        fig     = plt.figure(figsize = (width, height), constrained_layout=False)
        spec    = gridspec.GridSpec(ncols=2, nrows=n_axes, figure=fig,
                                height_ratios=heights,
                                width_ratios=[0.05, 0.95],
                                hspace=0.0, wspace=0.0)
        ax      = fig.add_subplot(spec[:, 0], frameon=False)
        fig_ax1 = fig.add_subplot(spec[0, 1])
        axes_list = []
        for i in channels:
            if offsets:
                line_label = f'offset {offsets[i]:+,}'
                fig_ax1.plot(   x, y[:,i] + offsets[i], c = self.colours[i],
                                drawstyle='steps-mid', linewidth = 0.4,
                                label = line_label)
                fig_ax1.plot(x, y_fit[:,i] + offsets[i], 'k', linewidth = 0.4)
            else:
                fig_ax1.plot(   x, y[:,i], c = self.colours[i],
                                drawstyle='steps-mid', linewidth = 0.4)
                fig_ax1.plot(x, y_fit[:,i], 'k', linewidth = 0.4)
                #, label = plot_legend)

            axes_list.append(fig.add_subplot(spec[i+1, 1]))
            difference = y[:,i] - y_fit[:,i]
            axes_list[i].plot(  x, difference, c = self.colours[i],
                                drawstyle='steps-mid',  linewidth = 0.4)
            if y_res_fit is not None:
                axes_list[i].plot(  x, y_res_fit[:,i], 'k:', linewidth = 0.4)
            axes_list[i].set_xticks(())
            tick = int(np.max(difference) * 0.67 / 100) * 100
            axes_list[i].set_yticks(([int(0), tick]))

        axes_list[-1].set_xlabel('time since trigger (s)')
        ax.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        ax.set_ylabel('counts / sec')
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(right=0.98)
        plt.subplots_adjust(top=0.98)
        plt.subplots_adjust(bottom=0.13)

        fig_ax1.ticklabel_format(axis = 'y', style = 'sci')
        fig_ax1.legend()

        plot_name = self.outdir + '/' + self.fstring + '_rates.pdf'
        if residuals is True:
            plot_name = self.outdir + '/' + self.fstring + '_residuals.pdf'
        fig.savefig(plot_name)


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


    # shitty_function()
    # Trigger = ResidualAnalysis(trigger = 999, times = (3.5, 5),
    #             datatype = 'discsc', nSamples = 500, sampler = SAMPLER,
    #             priors_pulse_start = 3, priors_pulse_end = 7)

    Trigger = ResidualAnalysis(trigger = 8099, times = (2, 12),
                datatype = 'discsc', nSamples = 500, sampler = SAMPLER,
                priors_pulse_start = 1, priors_pulse_end = 6)
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

# def sine_gaussian( times, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
#     return (sg_A * np.exp(- np.square((times - sg_t_0) / sg_tau)) *

#
# if __name__ == '__main__':
#     nSamples = 200
#     sampler  = 'nestle'
#     test = BilbyObject(trigger = 1, times = (-2, 50), test = True,
#                 datatype = 'discsc', nSamples = nSamples, sampler = sampler,
#                 priors_pulse_start = -5, priors_pulse_end = 50,
#                 priors_td_lo = 0,  priors_td_hi = 30)
#     test.inject_signal(residuals = 'Bessel')
