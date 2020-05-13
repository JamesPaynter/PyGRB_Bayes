import unittest

from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform

from PyGRB.backend.makepriors import MakePriors

class TestMakePriors(unittest.TestCase):

    def setUp(self):
    ## set up is down before the iteration of each class method
        self.priors_pulse_start = 0.0
        self.priors_pulse_end   = 1.0
        self.priors_td_lo       = 0.0
        self.priors_td_hi       = 0.8
        self.FRED_pulses   = []
        self.FREDx_pulses  = []
        self.residuals_sg  = []
        self.residuals_bes = []
        self.lens          = False
        self.channel       = 0

    def tearDown(self):
    ## tear down is done at the end of each iteration of the class methods
        del self.priors_pulse_start
        del self.priors_pulse_end
        del self.FRED_pulses
        del self.residuals_sg
        del self.lens

    def test_prior_dict(self):
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        priors = prior_object.priors
        self.assertIsInstance(priors, bilbyPriorDict)


    def test_3_FRED_priors(self):
        FRED_pulses  = [1, 2, 3]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        priors  = prior_object.priors
        keys    = prior_object.keys
        ## [*priors] makes a list of all the keys in the priors dict
        prior_keys = [*priors]
        for key in keys:
            self.assertIn(key, priors)
        for key in priors:
            self.assertIn(key, prior_keys)

    def test_3_FRED_constraints(self):
        ## not sure how to test the constraint function works properly
        ## but looking at it, it seems to be correct lol
        ## this tests that it works at least
        FRED_pulses = [1, 2, 3]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        priors  = prior_object.priors
        # need to remove constraint key from priors before sampling
        # no there is an error in the bilby code methinks
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['start_1_a'][i] <= sample['start_2_a'][i])
            self.assertTrue(0 <= sample['start_2_a'][i] <= sample['start_3_a'][i])

    def test_3_sg_priors(self):
        residuals_sg = [1, 2, 3]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        priors  = prior_object.priors
        keys    = prior_object.keys
        ## [*priors] makes a list of all the keys in the priors dict
        prior_keys = [*priors]
        for key in keys:
            self.assertIn(key, priors)
        for key in priors:
            self.assertIn(key, prior_keys)

    def test_3_sg_constraints(self):
        ## not sure how to test the constraint function works properly
        ## but looking at it, it seems to be correct lol
        ## this tests that it works at least
        residuals_sg = [1, 2, 3]
        FRED_pulses  = [1, 2, 3]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        priors  = prior_object.priors
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['res_begin_1_a'][i] <= sample['res_begin_2_a'][i])
            self.assertTrue(0 <= sample['res_begin_2_a'][i] <= sample['res_begin_3_a'][i])

    def test_lens(self):
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        self.assertEqual(self.lens, prior_object.lens)

    def test_background(self):
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        keys = prior_object.keys
        self.assertEqual(['background_a'], keys)

    def test_FRED_sg(self):
        FRED_pulses  = [1]
        residuals_sg = [1]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        keys = prior_object.keys
        key_list = ['background_a', 'start_1_a', 'scale_1_a', 'tau_1_a', 'sg_A_1_a',
                    'res_begin_1_a', 'sg_lambda_1_a', 'sg_omega_1_a', 'xi_1_a', 'sg_phi_1_a']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FREDx(self):
        FREDx_pulses = [1]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        keys = prior_object.keys
        key_list = ['background_a', 'start_1_a', 'scale_1_a', 'tau_1_a', 'xi_1_a',
                    'gamma_1_a', 'nu_1_a']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_bes(self):
        residuals_bes = [1]
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        keys = prior_object.keys
        key_list = ['background_a', 'bes_A_1_a',
                    'bes_Omega_1_a', 'bes_s_1_a', 'res_begin_1_a', 'bes_Delta_1_a']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FRED_lens(self):
        FRED_pulses  = [1]
        lens         = True
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = lens,
                                    channel = self.channel,
                                    priors_td_lo = self.priors_td_lo,
                                    priors_td_hi = self.priors_td_hi)
        keys = prior_object.keys
        key_list = ['background_a', 'start_1_a', 'scale_1_a', 'tau_1_a', 'xi_1_a',
                    'magnification_ratio', 'time_delay']
        # asserts that all keys in each list exist in both lists
        # ie that the lists contain the same keys only
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_bad_key(self):
        key = 'banana'
        prior_object = MakePriors(  self.priors_pulse_start,
                                    self.priors_pulse_end,
                                    count_FRED  = self.FRED_pulses,
                                    count_FREDx = self.FREDx_pulses,
                                    count_sg  = self.residuals_sg,
                                    count_bes = self.residuals_bes,
                                    lens = self.lens,
                                    channel = self.channel)
        prior_object.keys += key
        def test(self):
            with self.assertRaises(Exception) as context:
                prior_object.populate_priors()
            self.assertTrue('Key not found : {}'.format(key) in context.exception)

if __name__ == '__main__':
    unittest.main()
