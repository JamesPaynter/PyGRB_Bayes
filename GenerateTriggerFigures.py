from DynamicBilby import BilbyObject
import numpy as np
import bilby
import scipy.stats as stats
from scipy.stats import gaussian_kde


import matplotlib.pyplot    as plt

from matplotlib import rc

import corner


rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class TriggerPlots(BilbyObject):
    """docstring for Trigger3770Plots."""

    def __init__(self, range, **kwargs):
        super(TriggerPlots, self).__init__(**kwargs)
        self.range = range
    def plot_del_t_del_mu(self):
        ## need to figure out how to resize plot
        ## can pass in own axes but needs to be formatted correctly
        width = 3.321
        height = (3.321 / 1.8)
        fig, ax = plt.subplots( figsize = (width, height),
                                constrained_layout = True)
        fig2, ax2 = plt.subplots( figsize = (width, height),
                                constrained_layout = True)
        fig3, ax3 = plt.subplots( figsize = (width, height),
                                constrained_layout = True)
        labels = ['$\\Delta t$', '$\\Delta \\mu$']
        for i in range(4):
            result_label = self.fstring + '_result_' + self.clabels[i]
            open_result  = self.outdir + '/' + result_label +'_result.json'
            result = bilby.result.read_in_result(filename=open_result)

            defaults_kwargs = dict(
            bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color = self.colours[i],
            truth_color='tab:orange', #quantiles=[0.16, 0.84],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            plot_density=False, plot_datapoints=False, fill_contours=True,
            max_n_ticks=3, labels = labels,
            range = self.range)

            x = result.posterior['time_delay'].values
            y = result.posterior['magnification_ratio'].values
            # corner.hist2d(x,y, **defaults_kwargs, fig=fig)
            # y = np.clip(y, 0.00001, 1)
            # arr = self.draw_samples_and_return_mass_disp(x, y, 1000000)
            vels = self.draw_samples_and_return_vel_disp(x, y, 1000000)
            # arr = arr[arr > 0]
            # arr = arr[arr < 1e6]
            print('Channel %d density' % i)
            # density = gaussian_kde(arr)
            # xs = np.geomspace(min(arr), max(arr), 100)
            # density.covariance_factor = lambda : .01
            # density._compute_covariance()
            # ax2.plot(xs,density(xs), color = self.colours[i], linewidth = 0.2)
            # ax2.fill_between(xs,density(xs), color = self.colours[i], alpha = 0.2)
            # print('Channel %d hist' % i)
            bins=np.geomspace(min(vels), max(vels), 1000)
            ax3.hist(vels, bins = bins, density = True, color = self.colours[i], alpha = 0.2)

            ### or if you don't want the wings just use corner 2d hist
            ax.set_xlabel('$\\Delta t$ (s)')
            ax.set_ylabel('$\\Delta \\mu$')
        plt_string = self.outdir + '/' + self.fstring + '_delmudelt.pdf'
        # plt_string2= self.outdir + '/' + self.fstring + '_mass_dist.pdf'
        plt_string3= self.outdir + '/' + self.fstring + '_vel_dist.pdf'
        # fig.savefig(plt_string)
        # ax2.set_xlim(1e4, 1e7)
        ax2.set_xscale('log')
        ax2.set_xlabel('$(1+z_L)M$ $($M$_\\odot )$')
        ax2.set_ylabel('Density')
        ax2.tick_params(axis='y', which = 'both', left = False, right = False, labelleft = False)
        ax3.set_xscale('log')
        ax3.set_xlabel('Velocity dispersion, $\\sigma$ (m s$^{-1}$)')
        ax3.set_ylabel('Density')
        ax3.tick_params(axis='y', which = 'both', left = False, right = False, labelleft = False)
        # fig2.savefig(plt_string2)
        fig3.savefig(plt_string3)

    def draw_samples_and_return_mass(self, delt, delm, nSamples):
        c_cubed = 2.6944e25
        G       = 6.674e-11
        solarM  = 1.998e30
        delt_arr= np.random.choice(delt, nSamples)
        delm_arr= 1/ np.random.choice(delm, nSamples)
        point_m = 0.5 * c_cubed * delt_arr / G / (
                np.divide((delm_arr - 1), np.sqrt(delm_arr)) + np.log(delm_arr))

        return point_m / solarM

    def draw_samples_and_return_vel_disp(self, delt, delm, nSamples):
        c_fived = 2.42160617e42
        solarM  = 1.998e30
        delt_arr= np.random.choice(delt, nSamples)
        delm_arr= 1/ np.random.choice(delm, nSamples)
        f       = np.random.sample(nSamples)
        chi_src = np.random.sample(nSamples) * 3.086e26
        delt_max= delt_arr * (delm_arr + 1) / (delm_arr - 1)
        isotherm= c_fived * delt_max / (32 * np.pi **2) / (chi_src * (f - f**2))
        return isotherm ** 0.25


if __name__ == '__main__':
    Trigger = TriggerPlots(trigger = 3770, times = (-.1, 1),
                model = 'lens_model_1',
                range = [(0.37, 0.42), (0.50, 1.5)],
                datatype = 'tte', nSamples = 5000, sampler = 'nestle',
                priors_pulse_start = -.1, priors_pulse_end = 1.0,
                priors_td_lo = 0,  priors_td_hi = 1.0)

    # Trigger = TriggerPlots(trigger = 973, times = (-2, 50),
    #             model = 'lens_model_1',
    #             range = [(21, 22.7), (0.25, 0.52)],
    #             datatype = 'discsc', nSamples = 500, sampler = 'nestle',
    #             priors_pulse_start = -5, priors_pulse_end = 50,
    #             priors_td_lo = 0,  priors_td_hi = 1.0)

    # Trigger = TriggerPlots(trigger = 2571, times = (-10, 40),
    #             model = 'lens_model_1',
    #             # range = [(7.75, 8.03), (0.6, .9)],
    #             range = [(7.75, 8.45), (0.27, .9)],
    #             datatype = 'discsc', nSamples = 500, sampler = 'nestle',
    #             priors_pulse_start = -5, priors_pulse_end = 50,
    #             priors_td_lo = 0,  priors_td_hi = 1.0)

    Trigger.num_pulses = 1
    Trigger.tlabel     = Trigger.get_trigger_label()
    Trigger.fstring    = Trigger.get_file_string()
    Trigger.outdir     = Trigger.get_directory_name()
    Trigger.plot_del_t_del_mu()
