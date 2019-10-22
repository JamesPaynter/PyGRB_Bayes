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

    def inject_4_channel_lens(self):
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

        print(sample)

        times  = np.arange(800) * 0.064 - 2
        dt     = np.diff(times)
        t_0    = -2
        test_rates = self.one_FRED_rate(dt, t_0, **sample)


        ax.plot(times, rates, linewidth = 0.5, label = label[i])

        ax.set_xlim(times[0], times[-1])
        plt.locator_params(axis='y', nbins=4)
        plt.legend(frameon=False, loc = 1)
        plt.rcParams.update({'font.size': 8})
        plt.savefig('TestRates.pdf')



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

        sample['background'] = 3000 / 0.064
        sample['time_delay'] = 17
        sample['magnification_ratio'] = 0.4
        sample['start_1']    = 2
        sample['scale_1']    = 3e7
        sample['tau_1']      = 2
        sample['xi_1']       = 3

        times = np.arange(800) * 0.064 - 2 ## = 51.2
        dt = np.diff(times)
        t_0 = -2

        length = 10
        modes  = np.zeros(length)
        CI90   = np.zeros(length)
        CI99   = np.zeros(length)
        ## with bg as given 1e6 is a good minimumS
        scales = np.geomspace(1e8, 1e9, length)
        for scale in scales:
            sample['scale_1'] = scale
            test_rates = self.one_FRED_lens_rate(dt, t_0, **sample)
            BayesFactors = np.zeros(nSamples)
            for j in range(nSamples):
                noise_rates = np.random.poisson(test_rates)
                final_rates = np.zeros((len(times),1))
                final_rates[:,0] = noise_rates
                if j == 0:
                    plt.plot(times, test_rates)
                    plt.plot(times, noise_rates, c='k', alpha = 0.3)
                    plot_name = self.outdir + '/injected_signal'
                    plt.savefig(plot_name)
                self.GRB.bin_left, self.GRB.rates = times, final_rates
                self.GRB.bin_right = self.GRB.bin_left + 0.064
                widths = self.GRB.bin_right - self.GRB.bin_left
                self.GRB.counts = self.GRB.rates * widths
                evidences_2_FRED, errors_2_FRED = test.two_FRED(channels = [0], test = True)
                evidences_1_lens, errors_1_lens = test.one_FRED_lens(channels = [0], test = True)
                BayesFactors[j] = (evidences_1_lens[0] - evidences_2_FRED[0])

            [modeBayesFactor], _ = scipy.stats.mode(BayesFactors, axis=None)
            CI90BayesFactor = pymc3.stats.hpd(BayesFactors, alpha = 0.10)
            CI99BayesFactor = pymc3.stats.hpd(BayesFactors, alpha = 0.01)
            print(modeBayesFactor)
            print(CI90BayesFactor)
            print(CI99BayesFactor)

            plt.close('all')
            fig, ax = plt.subplots()
            ax.hist(BayesFactors)
            fig.savefig('plot.pdf')
            plt.close()
            break


if __name__ == '__main__':
    # test = TestPlots(trigger = 973, times = (-2,50), datatype = 'discsc',
    #                     sampler = 'Nestle')
    #
    # test.generate_figure_pulses()
    test = TestRecovery(trigger = 000, times = (-2, 50), test = True,
                datatype = 'discsc', nSamples = 100, sampler = 'Nestle',
                priors_pulse_start = -5, priors_pulse_end = 50,
                priors_td_lo = 0,  priors_td_hi = 30)

    test.get_recovery_plot(10)
