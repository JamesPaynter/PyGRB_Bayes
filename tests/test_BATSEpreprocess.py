import unittest

from PyGRB_Bayes.BATSEpreprocess import BATSESignal


class TestBATSESignal(unittest.TestCase):

    def setUp(self):
    # set up is down before the iteration of each class method
        self.burst    = 973
        self.datatype = 'discsc'
        self.times    = 'full'
        self.bgs      = True

    def tearDown(self):
    # tear down is done at the end of each iteration of a class method
        del self.burst
        del self.datatype
        del self.times
        del self.bgs

    def test_burst_assignment(self):
        # update these tests to ignore FileNotFoundError
        test = BATSESignal(self.burst,  datatype = self.datatype,
                                        times = self.times, bgs = self.bgs)

    def test_datatype_assignment(self):
        # update these tests to ignore FileNotFoundError
        test = BATSESignal(self.burst,  datatype = self.datatype,
                                        times = self.times, bgs = self.bgs)
        self.assertEqual(self.datatype, test.datatype)

    def test_times_assignment(self):
        # update these tests to ignore FileNotFoundError
        test = BATSESignal(self.burst,  datatype = self.datatype,
                                        times = self.times, bgs = self.bgs)
        self.assertEqual(self.times, test.times)

    # def test_bgs_assignment(self):
    #     # update these tests to ignore FileNotFoundError
    #     test = BATSESignal(self.burst,  datatype = self.datatype,
    #                                     times = self.times, bgs = self.bgs)
    #     self.assertEqual(self.bgs, test.bgs)

    def test_burst_type(self):
        burst = "banana"
        with self.assertRaises(ValueError):
            BATSESignal(burst,  datatype = self.datatype,
                                times = self.times, bgs = self.bgs)

    def test_datatype_not_string(self):
        datatype = 3.0
        with self.assertRaises(ValueError):
            BATSESignal(self.burst, datatype = datatype,
                                    times = self.times, bgs = self.bgs)

    def test_datatype_not_specified(self):
        datatype = 'orange'
        with self.assertRaises(AssertionError):
            BATSESignal(self.burst, datatype = datatype,
                                    times = self.times, bgs = self.bgs)

    def test_file_not_found(self):
        pass

    def test_BATSE_not_in_4B_T90_list(self):
        pass
