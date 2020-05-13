import sys, os
import argparse

from PyGRB.main.fitpulse import PulseFitter
from PyGRB.backend.makemodels import create_model_from_key
from PyGRB.backend.makemodels import make_two_pulse_models

import numpy as np
# status = 'print'
# status = 'ignore'
# np.seterr(divide=status, invalid=status)#, over = status)

def load_2151(sampler = 'dynesty', nSamples = 100):
    test = PulseFitter(2151, times = (-.1, 1.3),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 1,
                # the SG residual fits too late, need to edit pulse start time
                # priors for separate pulses
                priors_td_lo = 0,  priors_td_hi = 1)
    return test


def analysis_for_2151(indices):
    num_samples = [2000, 4500]
    for samples in num_samples:
        GRB = load_2151(sampler=SAMPLER, nSamples=samples)
        GRB.offsets = [0, 4000, 8000, -3000]
        GRB.test_two_pulse_models(indices, channels = [0, 1, 2, 3])


def evidence_for_2151():
    num_samples = [2000]
    # num_samples = [500, 2000, 4500]
    for samples in num_samples:
        GRB = load_2151(sampler=SAMPLER, nSamples=samples)
        GRB.offsets = [0, 4000, 8000, -3000]
        # model_dict = make_two_pulse_models()
        keys = ['FF', 'FL']#, 'XL']
        model_dict = {}
        for key in keys:
            model_dict[key] = create_model_from_key(key)
        models = [model for key, model in model_dict.items()]
        for model in models:
            # try:
            GRB.main_multi_channel(channels = [0, 1, 2, 3], model = model)
            # GRB.main_joint_multi_channel(channels = [0, 1, 2, 3], model = model)
            # GRB.get_residuals(channels = [0, 1, 2, 3], model = model)

            # lens_bounds = [(0.37, 0.42), (0.60, 1.8)]
            # GRB.lens_calc(model = model, lens_bounds = lens_bounds)
            # except:
            #     pass
        GRB.get_evidence_singular_lens()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(   description = 'Core bilby wrapper')
    parser.add_argument('--HPC', action = 'store_true',
                        help = 'Are you running this on SPARTAN ?')
    parser.add_argument('-i', '--indices', type=int, nargs='+',
                        help='an integer for indexing geomspace array')
    args = parser.parse_args()
    HPC = args.HPC


    if not HPC:
        from matplotlib import rc
        rc('font', **{'family': 'DejaVu Sans',
                    'serif': ['Computer Modern'],'size': 8})
        rc('text', usetex=True)
        rc('text.latex',
        preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}')
        SAMPLER = 'Nestle'
        evidence_for_2151()
        # analysis_for_2151([12])


    else:
        SAMPLER = 'dynesty'
        analysis_for_2151(args.indices)
