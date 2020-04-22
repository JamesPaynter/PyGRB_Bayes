# import numpy as np
# import pandas as pd
# from astropy.io import fits
#
#
#
# file = 'stte_list_3770.fits.gz'
# file = 'tte_bfits_3770.fits.gz'
# file = 'discsc_bfits_8099.fits.gz'
# # file = 'discsc_bfits_8099_2.fits.gz'
#
#
# # with fits.open(file) as hdu_list:
#     # for header in hdu_list[2].header:
#         # print(header)
#
#
# # data, header = fits.getdata("input_file.fits", header=True)
# a = fits.getheader(file, 2)
# print(a)
#
#
#
#
#


import sys, os
import numpy as np
import argparse
from PyGRB_Bayes.main.fitpulse import PulseFitter
from PyGRB_Bayes.main.visualisations import AnimateConvergence
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_one_pulse_models
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models

np.seterr(divide='ignore', invalid='ignore', over = 'ignore')

def load_0999(sampler = 'dynesty', nSamples = 2000):
    test = PulseFitter(999, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15,
                sampler_kwargs = {'n_check_point' : 200000000000})
    return test




if __name__ == '__main__':
    GRB = load_0999(sampler = 'dynesty')
