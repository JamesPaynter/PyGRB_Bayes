import unittest

from PyGRB_Bayes.core import DynamicBackEnd



class TestMakeKeys(unittest.TestCase):

    def setUp(self):
    # set up is down before the iteration of each class method
        self.FRED_pulses  = []
        self.residuals_sg = []
        self.lens         = False

    def tearDown(self):
    # tear down is done at the end of each iteration of a class method
        del self.FRED_pulses
        del self.residuals_sg
        del self.lens

    def test_lens(self):
        key_object   = DynamicBackEnd.MakeKeys( count_FRED = self.FRED_pulses,
                                                count_sg = self.residuals_sg,
                                                lens = self.lens)
        self.assertEqual(self.lens, key_object.lens)

    def test_background(self):
        key_object   = DynamicBackEnd.MakeKeys( count_FRED = self.FRED_pulses,
                                                count_sg = self.residuals_sg,
                                                lens = self.lens)
        keys = key_object.keys
        self.assertEqual(['background'], keys)

    def test_FRED(self):
        self.FRED_pulses  = [1]
        key_object   = DynamicBackEnd.MakeKeys( count_FRED = self.FRED_pulses,
                                                count_sg = self.residuals_sg,
                                                lens = self.lens)
        keys = key_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'xi_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_sg(self):
        self.residuals_sg = [1]
        key_object   = DynamicBackEnd.MakeKeys( count_FRED = self.FRED_pulses,
                                                count_sg = self.residuals_sg,
                                                lens = self.lens)
        keys = key_object.keys
        key_list = ['background', 'sg_A_1',
                    'res_begin_1', 'sg_lambda_1', 'sg_omega_1', 'sg_phi_1']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FRED_lens(self):
        self.FRED_pulses  = [1]
        self.lens         = True
        key_object   = DynamicBackEnd.MakeKeys( count_FRED = self.FRED_pulses,
                                                count_sg = self.residuals_sg,
                                                lens = self.lens)
        keys = key_object.keys
        key_list = ['background', 'start_1', 'scale_1', 'tau_1', 'xi_1',
                    'magnification_ratio', 'time_delay']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

if __name__ == '__main__':
    unittest.main()
