import unittest

from bilby.core.prior       import PriorDict        as bilbyPriorDict
from bilby.core.prior       import Uniform          as bilbyUniform
from bilby.core.prior       import Constraint       as bilbyConstraint
from bilby.core.prior       import LogUniform       as bilbyLogUniform
from bilby.core.prior       import DeltaFunction    as bilbyDeltaFunction

from PyGRB_Bayes.PyGRB_Bayes import DynamicBackEnd



class TestMakePriors(unittest.TestCase):

    def setUp(self):
    ## set up is down before the iteration of each class method
        self.priors_pulse_start = 0.0
        self.priors_pulse_end   = 1.0
        self.FRED_pulses  = []
        self.residuals_sg = []
        self.lens         = False

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
                                            count_FRED = self.FRED_pulses,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        priors = prior_object.priors
        self.assertIsInstance(priors, bilbyPriorDict)


    def test_3_FRED_priors(self):
        self.FRED_pulses  = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        priors  = prior_object.priors
        keys    = prior_object.keys
        ## [*priors] makes a list of all the keys in the priors dict
        prior_keys = [*priors] + ['constraint_2', 'constraint_3']
        for key in keys:
            self.assertIn(key, priors)
        for key in priors:
            self.assertIn(key, prior_keys)

    def test_3_FRED_constraints(self):
        ## not sure how to test the constraint function works properly
        ## but looking at it, it seems to be correct lol
        ## this tests that it works at least
        self.FRED_pulses = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            lens = self.lens)
        priors  = prior_object.priors
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['start_1'][i] <= sample['start_2'][i])
            self.assertTrue(0 <= sample['start_2'][i] <= sample['start_3'][i])

    def test_3_sg_priors(self):
        self.residuals_sg  = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        priors  = prior_object.priors
        keys    = prior_object.keys
        ## [*priors] makes a list of all the keys in the priors dict
        prior_keys = [*priors] + ['constraint_2_res', 'constraint_3_res']
        for key in keys:
            self.assertIn(key, priors)
        for key in priors:
            self.assertIn(key, prior_keys)

    def test_3_sg_constraints(self):
        ## not sure how to test the constraint function works properly
        ## but looking at it, it seems to be correct lol
        ## this tests that it works at least
        self.residuals_sg = [1, 2, 3]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        priors  = prior_object.priors
        sample  = priors.sample(100)
        for i in range(100):
            self.assertTrue(0 <= sample['res_begin_1'][i] <= sample['res_begin_2'][i])
            self.assertTrue(0 <= sample['res_begin_2'][i] <= sample['res_begin_3'][i])

    def test_lens(self):
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        self.assertEqual(self.lens, prior_object.lens)

    def test_background(self):
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            count_sg = self.residuals_sg,
                                            lens = self.lens)
        keys = prior_object.keys
        self.assertEqual(['background'], keys)

    def test_FRED_sg(self):
        self.FRED_pulses  = [1]
        self.residuals_sg = [1]
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            count_sg = self.residuals_sg,
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
                                            count_FREDx = FREDx_pulses)
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
                                            count_bes = residuals_bes)
        keys = prior_object.keys
        key_list = ['background', 'bes_A_1',
                    'bes_Omega_1', 'bes_s_1', 'res_begin_1', 'bes_Delta_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FRED_lens(self):
        self.FRED_pulses  = [1]
        self.lens         = True
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                            self.priors_pulse_end,
                                            count_FRED = self.FRED_pulses,
                                            lens = self.lens)
        keys = prior_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'xi_1',
                    'magnification_ratio', 'time_delay']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_bad_key(self):
        key = 'banana'
        prior_object = DynamicBackEnd.MakePriors(self.priors_pulse_start,
                                                 self.priors_pulse_end)
        prior_object.keys += key

        def test(self):
            with self.assertRaises(Exception) as context:
                prior_object.populate_priors()
            self.assertTrue('Key not found : {}'.format(key) in context.exception)

if __name__ == '__main__':
    unittest.main()
