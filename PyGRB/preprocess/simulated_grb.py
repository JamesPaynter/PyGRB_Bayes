import numpy as np
from PyGRB.preprocess.grb import EmptyGRB

GRB_discsc = EmptyGRB(  bin_left  = np.arange(100) * 0.064,
                        bin_right = np.arange(100) * 0.064 + 0.064,
                        counts    = np.ones((100,1)),
                        burst     = 1,
                        colours   = ['g'],
                        clabels   = ['1'],
                        datatype  = 'discsc',
                        satellite = 'test',
                        simulated = True)

GRB_tte    = EmptyGRB(  bin_left  = np.arange(100) * 0.064,
                        bin_right = np.arange(100) * 0.005 + 0.005,
                        counts    = np.ones((100,1)),
                        burst     = 1,
                        colours   = ['g'],
                        clabels   = ['1'],
                        datatype  = 'tte',
                        satellite = 'test',
                        simulated = True)
