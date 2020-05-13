import unittest
import numpy as np

from PyGRB.preprocess.grb import EmptyGRB


class TestEmptyGRB(unittest.TestCase):

    def setUp(self):
    # set up is down before the iteration of each class method
        self.bin_left  = np.arange(10)
        self.bin_right = np.arange(10) + 1
        self.counts    = np.ones((10,4))

    def tearDown(self):
    # tear down is done at the end of each iteration of a class method
        del self.bin_left
        del self.bin_right
        del self.counts

    def test_unequal_time_bins(self):
        bin_left = np.arange(9)
        with self.assertRaises(AssertionError):
            EmptyGRB(bin_left = bin_left, bin_right = self.bin_right,
                        counts = self.counts)

    def test_unequal_time_counts(self):
        counts = np.ones((9,4))
        with self.assertRaises(AssertionError):
            EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                        counts = counts)

    def test_bin_cross_over(self):
        bin_left  = np.arange(10) - 0.1
        with self.assertRaises(AssertionError):
            EmptyGRB(bin_left = bin_left, bin_right = self.bin_right,
                        counts = self.counts)

    def test_burst_assignment(self):
        test = EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                           counts = self.counts)

    def test_bin_left_assignment(self):
        test = EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                           counts = self.counts)
        self.assertTrue(np.array_equal(self.bin_left, test.bin_left))

    def test_bin_right_assignment(self):
        test = EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                           counts = self.counts)
        self.assertTrue(np.array_equal(self.bin_right, test.bin_right))

    def test_counts_assignment(self):
        test = EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                           counts = self.counts)
        self.assertTrue(np.array_equal(self.counts, test.counts))

    def test_assert_type_error_bin_left(self):
        bin_left = 'crocodile'
        with self.assertRaises(ValueError):
            EmptyGRB(bin_left = bin_left, bin_right = self.bin_right,
                        counts = self.counts)

    def test_assert_type_error_bin_right(self):
        bin_right = 2j
        with self.assertRaises(ValueError):
            EmptyGRB(bin_left = self.bin_left, bin_right = bin_right,
                        counts = self.counts)

    def test_assert_type_error_counts(self):
        counts = ['peanut']
        with self.assertRaises(ValueError):
            EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                        counts = counts)

    def test_counts_shape(self):
        counts = np.ones((4,10))
        with self.assertRaises(AssertionError):
            EmptyGRB(bin_left = self.bin_left, bin_right = self.bin_right,
                        counts = counts)
