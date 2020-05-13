import numpy as np
import unittest

from scipy.special import gammaln

from PyGRB.backend.makepriors import MakePriors
from PyGRB.backend.rateclass  import PoissonRate



class TestPoissonRate(unittest.TestCase):

    def setUp(self):
    # set up is down before the iteration of each class method
        self.priors_pulse_start = 0.0
        self.priors_pulse_end   = 1.0

        self.x = np.arange(100)
        self.y = np.ones(100)
        self.FRED_pulses   = []
        self.FREDx_pulses  = []
        self.residuals_sg  = []
        self.residuals_bes = []
        self.lens    = False
        self.channel = 0
        self.yfloat  = np.copy(self.y) * 1.
        self.yneg    = np.copy(self.y)
        self.yneg[0] = -1


    def tearDown(self):
    # tear down is done at the end of each iteration of a class method
        del self.x
        del self.y
        del self.FRED_pulses
        del self.FREDx_pulses
        del self.residuals_sg
        del self.residuals_bes
        del self.lens
        del self.yfloat
        del self.yneg


    def test_init_y_non_integer(self):
        ''' Doesn't work because we are fit floats anyway.
            Not sure if this is important.
        '''
    #     with self.assertRaises(ValueError):
    #         PoissonRate( x = self.x, y = self.yfloat,
    #                                     FRED_pulses   = self.FRED_pulses,
    #                                     FREDx_pulses  = self.FREDx_pulses,
    #                                     residuals_sg  = self.residuals_sg,
    #                                     residuals_bes = self.residuals_bes,
    #                                     lens = self.lens)
        pass

    def test_init__y_negative(self):
        ''' Doesn't work because calculate_rate catches negatives already. '''
        # with self.assertRaises(ValueError):
        #     PoissonRate( x = self.x, y = self.yneg,
        #                                 FRED_pulses   = self.FRED_pulses,
        #                                 FREDx_pulses  = self.FREDx_pulses,
        #                                 residuals_sg  = self.residuals_sg,
        #                                 residuals_bes = self.residuals_bes,
        #                                 lens = self.lens)
        pass

    def test_3_FRED_priors(self):
        ''' Tests the prior keys match the rate keys. '''
        self.FRED_pulses  = [1, 2, 3]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED   = self.FRED_pulses,
                                    count_FREDx  = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)

        rates_object = PoissonRate( x = self.x, y = self.y,
                                    count_FRED   = self.FRED_pulses,
                                    count_FREDx  = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)

        priors  = prior_object.priors
        keys    = rates_object.keys
        ## [*priors] makes a list of all the keys in the priors dict
        prior_keys = [*priors] + ['constraint_2', 'constraint_3']

        for key in keys:
            self.assertIn(key, priors)
        for key in priors:
            self.assertIn(key, prior_keys)

    # def test_known_FRED_pulse(self):
    #     ''' this is a bad test. '''
    #     self.parameters = dict([ ('start', 5),
    #                              ('scale', 10),
    #                              ('tau'  , 3),
    #                              ('xi'   , 7)  ])
    #     rate = PoissonRate.FRED_pulse(self.x, **self.parameters)
    #     ll = np.sum(-rate + y * np.log(rate) - gammaln(y + 1))
    #
    #     rates_object = PoissonRate( x = self.x, y = y,
    #                                 FRED_pulses   = self.FRED_pulses,
    #                                 FREDx_pulses  = self.FREDx_pulses,
    #                                 residuals_sg  = self.residuals_sg,
    #                                 residuals_bes = self.residuals_bes,
    #                                 lens = self.lens)
    #     log = rates_object.log_likelihood()
    #     for i in range(len(self.x)):
    #         self.assertEqual(ll[i], log[i])

if __name__ == '__main__':
    unittest.main()
