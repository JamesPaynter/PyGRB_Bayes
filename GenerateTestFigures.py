from DynamicBilby import BilbyObject
import numpy as np
import matplotlib.pyplot    as plt

from matplotlib import rc

import pymc3
import scipy.stats


rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class TestPlots(BilbyObject):
    """docstring for TestPlots."""

    def __init__(self, **kwargs):
        super(TestPlots, self).__init__(**kwargs)

    def generate_figure_pulses(self):
        width  = 3.321
        height = 3.321 / 1.6
        fig, ax = plt.subplots(figsize = (width, height), constrained_layout=True)

        sc = [1e4, 3.32e4, 2.18e12]
        xi = [0.4, 1, 10]
        ta = [2, 2, 2, 2]
        xk = ['k:', 'k--', 'k', 'k:']
        label = [   '$\\xi = 0.4$,  $A=10^{4}$',
                    '$\\xi = 1.0$,  $A=3.32\\times 10^{4}$ ',
                    '$\\xi = 10$,   $A=2.18\\times 10^{12}$ ']
        for i in range(3):
            times, rates = self.inject_FRED_signal( bg = 50,  st = 2,
                                                    sc = sc[i], ta = ta[i],
                                                    xi = xi[i])
            ax.plot(times, rates, xk[i], linewidth = 0.5, label = label[i])

        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('time (s)')
        ax.set_ylabel('counts')
        plt.locator_params(axis='y', nbins=4)
        plt.legend(frameon=False, loc = 1)
        plt.rcParams.update({'font.size': 8})
        plt.savefig('TestRates.pdf')

    def inject_FRED_signal(self, bg, st, sc, ta, xi):
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = None)
        sample = self.priors.sample()
        sample['background']            = bg
        sample['start_1']               = st
        sample['scale_1']               = sc
        sample['tau_1']                 = ta
        sample['xi_1']                  = xi

        print(sample)

        times  = np.arange(800) * 0.064 - 2
        dt     = np.diff(times)
        t_0    = -2
        test_rates = self.one_FRED_rate(dt, t_0, **sample)

        return times, test_rates

    def inject_4_channel_lens(self, bg, st, sc, ta, xi):
        width  = 3.321
        height = 3.321 / 1.6
        fig, ax = plt.subplots(figsize = (width, height), constrained_layout=True)

        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = False,
                            constraint = None)
        sample = self.priors.sample()
        sample['background']            = bg
        sample['start_1']               = st
        sample['scale_1']               = sc
        sample['tau_1']                 = ta
        sample['xi_1']                  = xi

        times  = np.arange(800) * 0.064 - 2
        dt     = np.diff(times)
        t_0    = -2

        scales = np.array([0.5, 0.7, 1.0, 0.3]) * sample['scale_1']
        for i in range(4):
            sample['tau_1'] *= 0.8
            sample['scale_1'] = scales[i]
            counts = self.one_FRED_rate(dt, t_0, **sample)
            noise_counts = np.random.poisson(counts)
            rates = noise_counts / 0.064
            ax.plot(times, rates, linewidth = 0.5, c = self.colours[i])

        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('time (s)')
        ax.set_ylabel('counts / sec')
        plt.locator_params(axis='y', nbins=4)
        # plt.legend(frameon=False, loc = 1)
        plt.rcParams.update({'font.size': 8})
        plt.savefig('Test4ChannelRates.pdf')



class TestRecovery(BilbyObject):
    """docstring for TestRecovery."""

    def __init__(self, **kwargs):
        super(TestRecovery, self).__init__(**kwargs)

    def get_recovery_plot(self, nSamples):
        self.model  = 'one FRED pulse lens'
        self.num_pulses = 1
        self.make_priors(   FRED = [1], FREDx = None,
                            gaussian = None, lens = True,
                            constraint = None)
        sample = self.priors.sample()

        sample['background'] = 3000
        sample['time_delay'] = 17
        sample['magnification_ratio'] = 0.4
        sample['start_1']    = 2
        sample['scale_1']    = 3e7
        sample['tau_1']      = 2
        sample['xi_1']       = 3

        t_0 = -2
        bin_size = 0.064
        times = np.arange(800) * bin_size + t_0 ## = 51.2
        dt = np.diff(times)

        self.counter = 0
        length = 10
        modes  = np.zeros(length)
        CI90   = np.zeros(length)
        CI99   = np.zeros(length)
        ## with bg as given 1e6 is a good minimum Scale
        scales = np.geomspace(1e7, 1e8, length)[::-1]
        for k in range(length):
            sample['scale_1'] = scales[k]
            test_counts  = self.one_FRED_lens_rate(dt, t_0, **sample)
            BayesFactors = np.zeros(nSamples)
            for j in range(nSamples):
                noise_counts = np.random.poisson(test_counts)
                final_counts = np.zeros((len(times),1))
                final_counts[:,0] = noise_counts
                if j == 0:
                    plt.plot(times, test_counts)
                    plt.plot(times, noise_counts, c='k', alpha = 0.3)
                    plot_name = self.outdir + '/injected_signal'
                    plt.savefig(plot_name)
                self.GRB.bin_left, self.GRB.counts = times, final_counts
                self.GRB.bin_right = self.GRB.bin_left + bin_size
                widths = self.GRB.bin_right - self.GRB.bin_left
                self.GRB.rates = self.GRB.counts / widths
                evidences_2_FRED, errors_2_FRED = test.two_FRED(
                                channels = [0], test = True, save_all = True)
                evidences_1_lens, errors_1_lens = test.one_FRED_lens(
                                channels = [0], test = True, save_all = True)
                BayesFactors[j] = (evidences_1_lens[0] - evidences_2_FRED[0])

            [modeBayesFactor], _ = scipy.stats.mode(BayesFactors, axis=None)
            CI90BayesFactor = pymc3.stats.hpd(BayesFactors, alpha = 0.10)
            CI99BayesFactor = pymc3.stats.hpd(BayesFactors, alpha = 0.01)
            print(modeBayesFactor)
            print(CI90BayesFactor)
            print(CI99BayesFactor)

            modes[k] = modeBayesFactor
            CI90[k]  = CI90BayesFactor
            CI99[k]  = CI99BayesFactor
            plt.close('all')
            fig, ax = plt.subplots()
            ax.hist(BayesFactors)
            fig.savefig('plot.pdf')
            plt.close()
            break



if __name__ == '__main__':
    # test = TestPlots(trigger = 973, times = (-2,50), datatype = 'discsc',
    #                     sampler = 'Nestle',
    #                     priors_pulse_start = -5, priors_pulse_end = 50,)
    #
    # test.generate_figure_pulses()
    # test.inject_4_channel_lens( bg = 3000,  st = 2, sc = 3e4,
    #                             ta = 6, xi = 0.6)
    test = TestRecovery(trigger = 000, times = (-2, 50), test = True,
                datatype = 'discsc', nSamples = 100, sampler = 'Nestle',
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)

    test.get_recovery_plot(10)
