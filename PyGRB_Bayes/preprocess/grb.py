import numpy as np


class EmptyGRB(object):
    """ EmptyGRB for Bilby signal injections. """

    def __init__(self, bin_left, bin_right, rates):
        """
        Initialize the :class:`~SignalFramework` abstract class. This class
        should be inherited by each Satellite's child class and the init ran
        after the init of the child classes.

        Parameters
        ----------
        bin_left : np.array.
            The parameter specifies the left bins of the GRB.

        bin_left : np.array.
            The parameter specifies the right bins of the GRB.

        rates : np.array.
            The parameter specifies rates at each bin of the GRB.
            If multi-channel, it should be in the form
        """
        super(EmptyGRB, self).__init__()

        if not isinstance(bin_left, np.ndarray):
            raise ValueError(
                'Input variable `bin_left` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(bin_left)))

        if not isinstance(bin_right, np.ndarray):
            raise ValueError(
                'Input variable `bin_right` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(bin_right)))

        if not isinstance(rates, np.ndarray):
            raise ValueError(
                'Input variable `rates` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(rates)))

        # assert right and left bin arrays are equal length
        assert(len(bin_left) == len(bin_right))
        # assert rates array is also the same length
        assert(len(bin_left)) == max(np.shape(rates))
        # assert that each left bin begins after the last right bin finishes
        assert(((bin_left[1:] - bin_right[:-1]) >= -1e-3).all())
        # assert rates has the right shape
        try:
            (a,b) = np.shape(rates)
        except:
            a, b = 1, 0
        assert(a > b)

        self.bin_left  = bin_left
        self.bin_right = bin_right
        self.rates     = rates
