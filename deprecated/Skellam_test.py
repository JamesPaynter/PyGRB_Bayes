import unittest
import numpy as np

from skellam_likelihood import SkellamLikelihood

import matplotlib.pyplot as plt
import scipy.special as special

class TestSkellamLikelihood(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.mu = 5
        self.x = np.linspace(0, 1, self.N)
        self.y_1 = np.random.poisson(self.mu, self.N)
        self.y_2 = np.random.poisson(self.mu, self.N)
        self.y_1float = np.copy(self.y_1) * 1.
        self.y_2float = np.copy(self.y_2) * 1.
        self.y_1neg = np.copy(self.y_1)
        self.y_2neg = np.copy(self.y_2)
        self.y_1neg[0] = -1
        self.y_2neg[0] = -1

        def test_function(x, c):
            return c

        def test_function_array(x, c):
            return np.ones(len(x)) * c

        self.function = test_function
        self.function_array = test_function_array
        self.skellam_likelihood = SkellamLikelihood(self.x, self.y_1,
                                                    self.y_2, self.function)
    def tearDown(self):
        del self.N
        del self.mu
        del self.x
        del self.y_1
        del self.y_2
        del self.y_1float
        del self.y_2float
        del self.y_1neg
        del self.y_2neg
        del self.function
        del self.function_array
        del self.skellam_likelihood

    def test_init_y_non_integer(self):
        with self.assertRaises(ValueError):
            SkellamLikelihood(self.x, self.y_1float, self.y_2float, self.function)

    def test_init__y_negative(self):
        with self.assertRaises(ValueError):
            SkellamLikelihood(self.x, self.y_1neg, self.y_2neg, self.function)

    def test_neg_rate(self):
        ''' Skellam can be negative so this test is not relevant. '''
        # self.skellam_likelihood.parameters['c'] = -2
        # with self.assertRaises(ValueError):
        #     self.skellam_likelihood.log_likelihood()
        pass

    def test_neg_rate_array(self):
        ''' Skellam can be negative so this test is not relevant. '''
        # likelihood = SkellamLikelihood( self.x, self.y_1,
        #                                 self.y_2, self.function_array)
        # likelihood.parameters['c'] = -2
        # print('*****')
        # print(type(likelihood.parameters['c']))
        # with self.assertRaises(ValueError):
        #     likelihood.log_likelihood()
        pass

    def test_init_y_1(self):
        self.assertTrue(np.array_equal(self.y_1, self.skellam_likelihood.y_1))

    def test_init_y_2(self):
        self.assertTrue(np.array_equal(self.y_2, self.skellam_likelihood.y_2))

    def test_set_y_1_to_array(self):
        new_y = np.arange(start=0, stop=50, step=2)
        self.skellam_likelihood.y_1 = new_y
        self.assertTrue(np.array_equal(new_y, self.skellam_likelihood.y_1))

    def test_set_y_2_to_array(self):
        new_y = np.arange(start=0, stop=50, step=2)
        self.skellam_likelihood.y_2 = new_y
        self.assertTrue(np.array_equal(new_y, self.skellam_likelihood.y_2))

    def test_set_y_1_to_positive_int(self):
        new_y = 5
        self.skellam_likelihood.y_1 = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.skellam_likelihood.y_1))

    def test_set_y_2_to_positive_int(self):
        new_y = 5
        self.skellam_likelihood.y_2 = new_y
        expected_y = np.array([new_y])
        self.assertTrue(np.array_equal(expected_y, self.skellam_likelihood.y_2))

    def test_set_y_1_to_negative_int(self):
        with self.assertRaises(ValueError):
            self.skellam_likelihood.y_1 = -5

    def test_set_y_2_to_negative_int(self):
        with self.assertRaises(ValueError):
            self.skellam_likelihood.y_2 = -5

    def test_set_y_1_to_float(self):
        with self.assertRaises(ValueError):
            self.skellam_likelihood.y_1 = 5.3

    def test_set_y_2_to_float(self):
        with self.assertRaises(ValueError):
            self.skellam_likelihood.y_2 = 5.3

    def test_log_likelihood_wrong_func_return_type(self):
        skellam_likelihood = SkellamLikelihood(
        x=self.x, y_1=self.y_1, y_2=self.y_2, func=lambda x: 'test')
        with self.assertRaises(ValueError):
            skellam_likelihood.log_likelihood()

    def test_log_likelihood_negative_func_return_element(self):
        ''' Skellam can be negative so this test is not relevant. '''
        # skellam_likelihood = SkellamLikelihood(x=self.x, y_1=self.y_1,
        #             y_2=self.y_2, func = lambda x: np.array([3, 6, -2]))
        # with self.assertRaises(ValueError):
        #     skellam_likelihood.log_likelihood()
        pass

    def test_log_likelihood_zero_func_return_element(self):
        ''' Skellam can be zero so this test is not relevant. '''
        # skellam_likelihood = SkellamLikelihood(x=self.x, y_1=self.y_1,
        #             y_2=self.y_2, func=lambda x: np.array([3, 6, 0]))
        # self.assertEqual(-np.inf, skellam_likelihood.log_likelihood())
        pass

    def test_repr(self):
        ''' __repr__ is cooked '''
    #     likelihood = SkellamLikelihood(
    #         self.x, self.y_1, self.y_2, self.function)
    #     expected = 'SkellamLikelihood(x={}, y_1={}, y_2={}, func={})'.format(
    #     self.x, self.y_1, self.y_2, self.function.__name__)
    #     self.assertEqual(expected, repr(likelihood))
        pass




class TestSkelzaaaa(object):
    def __init__(self):
        self.N = 100
        self.mu = 20
        self.x = np.linspace(0, 10, self.N+1)
        self.y_1 = np.random.poisson(self.mu, self.N)
        # self.y_2 = np.random.poisson(self.mu, self.N)

        deltat = np.diff(self.x)
        def test_function(x, c):
            return c

        def test_function_array(x, c):
            return np.ones(len(x)) * c

        def sine_gaussian(times, sg_A, sg_t_0, sg_tau, sg_omega, sg_phi):
            return (sg_A * np.exp(- np.square((times - sg_t_0) / sg_tau)) *
                    np.cos(sg_omega * times + sg_phi) )

        def residuals_bessel(times, bes_A, bes_Omega, bes_s, bes_t_0, bes_Delta):
            def inner_function(times, bes_A, bes_Omega, bes_s, bes_t_0, bes_Delta):
                if times > bes_t_0 + bes_Delta / 2:
                    return bes_A * special.j0(bes_s * bes_Omega *
                            (- bes_t_0 + times - bes_Delta / 2) )
                    # return times > bes_t_0 + bes_Delta / 2

                elif times < bes_t_0 - bes_Delta / 2:
                    return bes_A * special.j0(bes_Omega *
                            (  bes_t_0 - times - bes_Delta / 2) )
                    # return times < bes_t_0 - bes_Delta / 2
                else:
                    return bes_A
            vfunc = np.vectorize(inner_function)
            return vfunc(times, bes_A, bes_Omega, bes_s, bes_t_0, bes_Delta)

            # return np.where(times > bes_t_0 + bes_Delta / 2,
            #         bes_A * special.j0(bes_s * bes_Omega *
            #        (- bes_t_0 + times - bes_Delta / 2) ),
            #        (np.where(times < bes_t_0 - bes_Delta / 2,
            #         bes_A * special.j0(bes_Omega *
            #        (bes_t_0 - times - bes_Delta / 2) ),
            #        bes_A)))

        keys = dict()
        # keys['sg_A']    = 30
        # keys['sg_t_0']  = 2
        # keys['sg_tau']  = 2
        # keys['sg_omega']= 2
        # keys['sg_phi']  = 2

        keys['bes_A']       = 4
        keys['bes_Omega']   = 2
        keys['bes_s']       = 2
        keys['bes_t_0']     = 2
        keys['bes_Delta']   = 2

        self.function = residuals_bessel
        # self.y_2 = np.random.poisson(self.mu, self.N) + (
        self.y_2 = self.function(self.x[0:-1], **keys)#.astype('uint')
        print(self.y_2)
        self.skellam_likelihood = SkellamLikelihood(deltat, self.y_1,
                                                self.y_2, self.function)
        print('\n\n\n\n\n')
        self.skellam_likelihood.parameters = keys.copy()
        print(self.skellam_likelihood.log_likelihood())
        print('\n\n\n\n\n')


        fig, ax = plt.subplots(2)
        ax[0].plot(self.x[0:-1], self.y_1)
        ax[0].plot(self.x[0:-1], self.y_2)
        ax[1].plot(self.x[0:-1], self.y_1 - self.y_2, 'k:')
        fig.savefig('skellam.png')



if __name__ == '__main__':
    # unittest.main()
    aaa = TestSkelzaaaa()


    # self.x = np.ones(8)
    # self.y_1 = np.array([1, 2, 3, 1, 2, 1, 2, 3])
    # self.y_2 = np.array([4, 1, 1, 2, 2, 3, 2, 3])
    #
    # def model(x, c):
    #     return c
    #
    # aaa = SkellamLikelihood(x, y_1, y_2, model)
    # aaa.parameters['c'] = 5
    # aaa.log_likelihood()
