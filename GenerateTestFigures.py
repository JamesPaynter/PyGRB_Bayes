from DynamicBilby import BilbyObject
import numpy as np
import matplotlib.pyplot    as plt

class TestPlots(BilbyObject):
    """docstring for TestPlots."""

    def __init__(self, **kwargs):
        super(TestPlots, self).__init__(**kwargs)

    def generate_figure_pulses(self):
        width  = 3.321
        height = 3.321 / 2
        fig, ax = plt.subplots(figsize = (width, height), constrained_layout=True)

        sc = [1e4, 1e5, 1e6, 1e7]
        xi = [3, 3, 2, 4]
        ta = [2, 8, 2, 2]
        xk = ['k', 'k:', 'k-.', 'k--']
        for i in range(4):
            times, rates = self.inject_FRED_signal( bg = 50,  st = 2,
                                                    sc = sc[i], ta = ta[i],
                                                    xi = 3)
            ax.plot(times, rates, xk[i], linewidth = 0.5)

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


if __name__ == '__main__':
    test = TestPlots(trigger = 973, times = (-2,50), datatype = 'discsc',
                        sampler = 'Nestle')

    test.generate_figure_pulses()
