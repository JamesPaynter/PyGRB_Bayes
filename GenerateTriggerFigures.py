from DynamicBilby import BilbyObject
import numpy as np
import bilby
import scipy.stats as stats

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
            corner.hist2d(x,y, **defaults_kwargs, fig=fig)

            ### or if you don't want the wings just use corner 2d hist
            ax.set_xlabel('$\\Delta t$ (s)')
            ax.set_ylabel('$\\Delta \\mu$')
            plt_string = self.outdir + '/' + self.fstring + '_delmudelt.pdf'
            fig.savefig(plt_string)






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
