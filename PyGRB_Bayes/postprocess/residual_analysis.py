import numpy as np


def autocorrelations(residuals):
    """ Do a test to see if they are Gaussian distributed? """
    acf  = np.correlate(residuals, residuals, mode='full')[len(residuals)-1:]
    acf2 = np.correlate(residuals**2, residuals**2, mode='full')[len(residuals)-1:]
    acf, acf2 = acf / np.max(acf), acf2 / np.max(acf2)
    return acf, acf2

def quantile_plot():
    """
    Test that the residuals are normally dsitributed.

    Is this what they should be... ?
    """
    pass


class ResidualAnalysis(object):
    """docstring for ResidualAnalysis."""

    def __init__(self):
        super(ResidualAnalysis, self).__init__()

    @staticmethod
    def quartile_plot(self):
        pass
