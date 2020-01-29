import unittest

from PyGRB_Bayes.core import DynamicBilby

import sys
print(sys.path)



class TestMakeKeys(unittest.TestCase):

    def setUp(self):
        self.FRED_pulses  = [1, 2]
        self.residuals_sg = [1, 2]
        self.lens         = False

    def tearDown(self):
        del self.FRED_pulses
        del self.residuals_sg

    def test_keys(self):
        key_object = DynamicBilby.MakeKeys( self.FRED_pulses,
                                            self.residuals_sg, self.lens)
        print(key_object.keys)


if __name__ == '__main__':
    unittest.main()
