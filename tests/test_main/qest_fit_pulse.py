import unittest
import shutil



from PyGRB.main.fitpulse import PulseFitter
from PyGRB.backend.rate_functions import *
from PyGRB.backend.makemodels import create_model_dict
from PyGRB.backend.makemodels import create_model_from_key
from PyGRB.preprocess.simulated_grb import GRB_discsc, GRB_tte
from PyGRB.postprocess.plot_grb import GRBPlotter


class TestFredFit(unittest.TestCase):

    def setUp(self):
        self.model = 'F'
        self.times = np.arange(100) * 0.064

        self.st = 1.0
        self.sc = 20
        self.ta = 1.0
        self.xi = 1.0
        self.bg = 5


        self.IP = dict(start = self.st, scale = self.sc, tau = self.ta, xi = self.xi)
        self.injection_parameters = dict(start_1_a = self.st, scale_1_a = self.sc,
        tau_1_a = self.ta, xi_1_a = self.xi, background_a = self.bg)
        self.GRB = GRB_discsc
        self.GRB.counts[:,0] = np.random.poisson(
                FRED_pulse(self.times, **self.IP) + self.bg)

        nSamples = 200
        self.discsc_fit = PulseFitter(0, times = (0, 6.4),
                datatype = 'self.discsc_fit', nSamples = nSamples, sampler = 'nestle',
                priors_pulse_start = 0, priors_pulse_end = 5,
                GRB = self.GRB, test = True,
                priors_td_lo = 0,  priors_td_hi = 1,
                injection_parameters = self.injection_parameters,
                save = True)

    def tearDown(self):
        del self.st
        del self.sc
        del self.ta
        del self.xi
        del self.bg
        del self.IP
        del self.GRB
        del self.model
        del self.times
        del self.injection_parameters

        # try:
        #     shutil.rmtree(self.discsc_fit.base_folder)
        # except OSError:
        #     pass

        pass


    # def test_one_fred(self):
    #     model = create_model_from_key(self.model)
    #     self.discsc_fit.models = dict(F = model)
    #     self.discsc_fit.main_1_channel(channel = 0, model = model)
    #     GRBPlotter.plot_grb(self.discsc_fit.GRB, [0], self.discsc_fit.outdir)

class TestFredXFit(unittest.TestCase):

    def setUp(self):
        self.model = 'X'
        self.times = np.arange(100) * 0.064

        self.st = 1.0
        self.sc = 1000
        self.ta = 1.0
        self.xi = 1.0
        self.ga = 1.0
        self.nu = 1.0
        self.bg = 5


        self.IP = dict( start = self.st,
                        scale = self.sc,
                        tau = self.ta,
                        xi = self.xi,
                        gamma = self.ga,
                        nu = self.nu)

        self.injection_parameters = dict(
                        background_a = self.bg,
                        start_1_a = self.st,
                        scale_1_a = self.sc,
                        tau_1_a = self.ta,
                        xi_1_a = self.xi,
                        gamma_1_a = self.ga,
                        nu_1_a = self.nu)

        self.GRB = GRB_discsc
        self.GRB.counts[:,0] = np.random.poisson(
                FREDx_pulse(self.times, **self.IP) + self.bg)

        nSamples = 300
        self.discsc_fit = PulseFitter(0, times = (0, 6.4),
                datatype = 'self.discsc_fit', nSamples = nSamples, sampler = 'nestle',
                priors_pulse_start = 0, priors_pulse_end = 5,
                GRB = self.GRB, test = True,
                priors_td_lo = 0,  priors_td_hi = 1,
                injection_parameters = self.injection_parameters,
                save = True)

    def tearDown(self):
        del self.st
        del self.sc
        del self.ta
        del self.xi
        del self.bg
        del self.nu
        del self.ga
        del self.IP
        del self.GRB
        del self.model
        del self.times
        del self.injection_parameters

        # try:
        #     shutil.rmtree(self.discsc_fit.base_folder)
        # except OSError:
        #     pass

        pass


    # def test_one_fredx(self):
    #     model = create_model_from_key(self.model)
    #     self.discsc_fit.models = dict(X = model)
    #     self.discsc_fit.main_1_channel(channel = 0, model = model)
    #     GRBPlotter.plot_grb(self.discsc_fit.GRB, [0], self.discsc_fit.outdir)


if __name__ == '__main__':
    unittest.main()
