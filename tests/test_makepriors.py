import unittest

from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction

from PyGRB_Bayes import DynamicBackEnd



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

    def tearDown(self):
    ## tear down is done at the end of each iteration of the class methods
        del self.priors_pulse_start
        del self.priors_pulse_end
        del self.FRED_pulses
        del self.residuals_sg
        del self.lens

    def test_prior_dict(self):
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        priors = prior_object.priors
        self.assertIsInstance(priors, bilbyPriorDict)


    def test_3_FRED_priors(self):
        FRED_pulses  = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
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
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        priors  = prior_object.priors
        # need to remove constraint key from priors before sampling
        # no there is an error in the bilby code methinks
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['start_1'][i] <= sample['start_2'][i])
            self.assertTrue(0 <= sample['start_2'][i] <= sample['start_3'][i])

    def test_3_sg_priors(self):
        residuals_sg  = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
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
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        priors  = prior_object.priors
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['res_begin_1'][i] <= sample['res_begin_2'][i])
            self.assertTrue(0 <= sample['res_begin_2'][i] <= sample['res_begin_3'][i])

    def test_lens(self):
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        self.assertEqual(self.lens, prior_object.lens)

    def test_background(self):
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        keys = prior_object.keys
        self.assertEqual(['background'], keys)

    def test_FRED_sg(self):
        FRED_pulses  = [1]
        residuals_sg = [1]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        keys = prior_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'sg_A_1',
                    'res_begin_1', 'sg_lambda_1', 'sg_omega_1', 'xi_1', 'sg_phi_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FREDx(self):
        FREDx_pulses = [1]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        keys = prior_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'xi_1',
                    'gamma_1', 'nu_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_bes(self):
        residuals_bes = [1]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = residuals_bes,
                                            lens = self.lens)
        keys = prior_object.keys
        key_list = ['background', 'bes_A_1',
                    'bes_Omega_1', 'bes_s_1', 'res_begin_1', 'bes_Delta_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FRED_lens(self):
        FRED_pulses  = [1]
        lens         = True
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = lens,
                                            priors_td_lo = self.priors_td_lo,
                                            priors_td_hi = self.priors_td_hi)
        keys = prior_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'xi_1',
                    'magnification_ratio', 'time_delay']
        # asserts that all keys in each list exist in both lists
        # ie that the lists contain the same keys only
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_bad_key(self):
        key = 'banana'
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED  = self.FRED_pulses,
                                            count_FREDx = self.FREDx_pulses,
                                            count_sg  = self.residuals_sg,
                                            count_bes = self.residuals_bes,
                                            lens = self.lens)
        prior_object.keys += key
        def test(self):
            with self.assertRaises(Exception) as context:
                prior_object.populate_priors()
            self.assertTrue('Key not found : {}'.format(key) in context.exception)

if __name__ == '__main__':
    unittest.main()
