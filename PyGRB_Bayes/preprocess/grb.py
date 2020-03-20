import numpy as np


class EmptyGRB(object):
    """ EmptyGRB for Bilby signal injections. """

    def __init__(self, bin_left, bin_right, counts, **kwargs):
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

        counts : np.array.
            The parameter specifies counts at each bin of the GRB.
            If multi-channel, it should be in the form [:,k], where k is the
            number of channels.
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

        if not isinstance(counts, np.ndarray):
            raise ValueError(
                'Input variable `counts` should be a numpy array. '
                'Is {} when it should be np.ndarray.'.format(type(counts)))

        # assert right and left bin arrays are equal length
        assert(len(bin_left) == len(bin_right))
        # assert counts array is also the same length
        assert(len(bin_left) == max(np.shape(counts)))
        # assert that each left bin begins after the last right bin finishes
        assert(((bin_left[1:] - bin_right[:-1]) >= -1e-3).all())
        # assert counts has the right shape
        try:
            (a,b) = np.shape(counts)
        except:
            a, b = 1, 0
        assert(a > b)

        self.bin_left  = bin_left
        self.bin_right = bin_right
        self.counts    = counts
        self.burst     = kwargs.get('burst')
        self.colours   = kwargs.get('colours')
        self.clabels   = kwargs.get('clabels')
        self.datatype  = kwargs.get('datatype')
        self.satellite = kwargs.get('satellite')

        self.simulated = kwargs.get('simulated')
        if self.simulated:
            self.counts = np.random.poisson(counts)
