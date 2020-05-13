import unittest
from PyGRB.backend.makekeys import MakeKeys

class TestMakeKeys(unittest.TestCase):

    def setUp(self):
    # set up is down before the iteration of each class method
        self.FRED_pulses   = []
        self.FREDx_pulses  = []
        self.residuals_sg  = []
        self.residuals_bes = []
        self.channel       = 0
        self.lens          = False

    def tearDown(self):
    # tear down is done at the end of each iteration of a class method
        del self.FRED_pulses
        del self.residuals_sg
        del self.lens

    def test_lens(self):
        key_object   = MakeKeys(count_FRED  = self.FRED_pulses,
                                count_FREDx = self.FREDx_pulses,
                                count_sg  = self.residuals_sg,
                                count_bes = self.residuals_bes,
                                lens = self.lens,
                                channel = self.channel)
        self.assertEqual(self.lens, key_object.lens)

    def test_background(self):
        key_object   = MakeKeys(count_FRED  = self.FRED_pulses,
                                count_FREDx = self.FREDx_pulses,
                                count_sg  = self.residuals_sg,
                                count_bes = self.residuals_bes,
                                lens = self.lens,
                                channel = self.channel)
        keys = key_object.keys
        self.assertEqual(['background_a'], keys)

    def test_FRED(self):
        FRED_pulses  = [1]
        key_object   = MakeKeys(count_FRED  = FRED_pulses,
                                count_FREDx = self.FREDx_pulses,
                                count_sg  = self.residuals_sg,
                                count_bes = self.residuals_bes,
                                lens = self.lens,
                                channel = self.channel)
        keys = key_object.keys
        key_list = ['background_a', 'start_1_a', 'scale_1_a', 'tau_1_a', 'xi_1_a']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_sg(self):
        residuals_sg = [1]
        key_object   = MakeKeys(count_FRED  = self.FRED_pulses,
                                count_FREDx = self.FREDx_pulses,
                                count_sg  = residuals_sg,
                                count_bes = self.residuals_bes,
                                lens = self.lens,
                                channel = self.channel)
        keys = key_object.keys
        key_list = ['background_a', 'sg_A_1_a',
                    'res_begin_1_a', 'sg_lambda_1_a', 'sg_omega_1_a', 'sg_phi_1_a']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

    def test_FRED_lens(self):
        FRED_pulses  = [1]
        lens         = True
        key_object   = MakeKeys(count_FRED  = FRED_pulses,
                                count_FREDx = self.FREDx_pulses,
                                count_sg  = self.residuals_sg,
                                count_bes = self.residuals_bes,
                                lens = lens,
                                channel = self.channel)
        keys = key_object.keys
        key_list = ['background_a', 'start_1_a', 'scale_1_a', 'tau_1_a', 'xi_1_a',
                    'magnification_ratio', 'time_delay']
        for key in keys:
            self.assertIn(key, key_list)
        for key in key_list:
            self.assertIn(key, keys)

if __name__ == '__main__':
    unittest.main()
