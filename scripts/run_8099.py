import sys, os

import argparse
from PyGRB_Bayes.main.fitpulse import PulseFitter
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_one_pulse_models
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models

def load_8099(sampler = 'dynesty', nSamples = 200):
    test = PulseFitter(8099, times = (2, 15),
                datatype = 'discsc', nSamples = nSamples, sampler = sampler,
                priors_pulse_start = 0, priors_pulse_end = 15)
    return test

def analysis_for_8099(indices):
    GRB = load_8099(sampler = SAMPLER, nSamples = 500)
    GRB.test_pulse_type(indices, channels = [0, 1, 2, 3])



def evidence_for_8099():
    GRB = load_8099(sampler = SAMPLER, nSamples = 200)
    GRB.models = make_one_pulse_models()
    models = [model for key, model in GRB.models.items()]
    for model in models:
        # try:
        GRB.main_joint_multi_channel(channels = [0, 1, 2, 3], model = model)
            # GRB.get_residuals(channels = [0, 1, 2, 3], model = model)
        # except:
            # pass
    GRB.get_evidence_singular()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(   description = 'Core wrapper')
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
        evidence_for_8099()
        # analysis_for_8099([12,13,14,15])
    else:
        SAMPLER = 'dynesty'
        analysis_for_8099(args.indices)
