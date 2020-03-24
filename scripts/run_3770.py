import sys, os
import argparse

from PyGRB_Bayes.main.fitpulse import PulseFitter
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models

import numpy as np
# status = 'print'
# status = 'ignore'
# np.seterr(divide=status, invalid=status)#, over = status)

def load_3770(sampler = 'dynesty', nSamples = 100):
    test = PulseFitter(3770, times = (-.1, 1),
                datatype = 'tte', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = -.1, priors_pulse_end = 0.6,
                priors_td_lo = 0,  priors_td_hi = 0.5)
    return test


def analysis_for_3770(indices):
    num_samples = [2000, 4500]
    for samples in num_samples:
        GRB = load_3770(sampler=SAMPLER, nSamples=samples)
        GRB.offsets = [0, 4000, 8000, -3000]
        GRB.test_two_pulse_models(indices, channels = [0, 1, 2, 3])


def evidence_for_3770():
    num_samples = [500, 2000, 4500]
    for samples in num_samples:
        GRB = load_3770(sampler=SAMPLER, nSamples=samples)
        GRB.offsets = [0, 4000, 8000, -3000]
        # keys = ['FF', 'FL', 'FbFb', 'FbL', 'XX', 'XL', 'XbXb', 'XbL']
        model_dict = make_two_pulse_models()
        models = [model for key, model in model_dict.items()]
        for model in models:
            try:
            # GRB.main_joint_multi_channel(channels = [0, 1, 2, 3], model = model)
                GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
            except:
                pass
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
        # evidence_for_3770()
        analysis_for_3770([12])


    else:
        SAMPLER = 'dynesty'
        analysis_for_3770(args.indices)
