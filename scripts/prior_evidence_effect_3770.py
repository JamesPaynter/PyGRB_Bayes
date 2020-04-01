import sys, os
import time
import argparse
import numpy as np

from bilby.core.prior import Uniform          as bilbyUniform
from bilby.core.prior import DeltaFunction    as bilbyDeltaFunction
from bilby.core.prior import Gaussian         as bilbyGaussian

from PyGRB_Bayes.main.fitpulse import PulseFitter
from PyGRB_Bayes.backend.makemodels import create_model_from_key
from PyGRB_Bayes.backend.makemodels import make_two_pulse_models






def run_log_flat_priors(    indices,
                            model_keys,
                            channels,
                            nSamples,
                            n_per_split,
                            test
                            ):

    directory_labels = [    'small_box_log_flat_nr',
                            'mid_box_log_flat_nr',
                            'large_box_log_flat_nr']

    prior_sets       = [{   'priors_gamma_min': 1e-1, ## generic
                            'priors_gamma_max': 1e1,
                            'priors_nu_min'   : 1e-1,
                            'priors_nu_max'   : 1e1},
                        {   'priors_gamma_min': 1e-2,
                            'priors_gamma_max': 1e2,
                            'priors_nu_min'   : 1e-2,
                            'priors_nu_max'   : 1e2},
                        {   'priors_gamma_min': 1e-3,
                            'priors_gamma_max': 1e3,
                            'priors_nu_min'   : 1e-3,
                            'priors_nu_max'   : 1e3   }]


    for ii, prior_set in enumerate(prior_sets):
        if not test:
            GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                        datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                        priors_pulse_start = -.1, priors_pulse_end = 0.6,
                        priors_td_lo = 0,  priors_td_hi = 0.5,
                        directory_label = directory_labels[ii],
                        **prior_set)

            GRB_wrap.offsets = [0, 4000, 8000, -3000]
        else:
            print('Currently running log-flat priors')
        model_dict = {}
        for key in model_keys:
            model_dict[key] = create_model_from_key(key,
                        custom_name = f'{key}_{directory_labels[ii]}')
        models = [model for key, model in model_dict.items()]

        indx = np.intersect1d(indices,
            np.arange(n_per_split * ii, n_per_split * (ii + 1))) % n_per_split

        ## given 2 models and 4 channels should be passed indx in [0-7]
        if not test:
            GRB_wrap._split_array_job_to_4_channels(models   = models,
                                                    indices  = indx,
                                                    channels = channels)
        else:
            print(indx)

def run_flat_priors(    indices,
                        model_keys,
                        channels,
                        nSamples,
                        n_per_split,
                        test
                        ):

    directory_labels = [    'small_box_flat_nr',
                            'mid_box_flat_nr',
                            'large_box_flat_nr']
    prior_sets       = [{   'priors_gamma_min': 1e-1, ## generic
                            'priors_gamma_max': 1e1,
                            'priors_nu_min'   : 1e-1,
                            'priors_nu_max'   : 1e1},
                        {   'priors_gamma_min': 1e-2,
                            'priors_gamma_max': 1e2,
                            'priors_nu_min'   : 1e-2,
                            'priors_nu_max'   : 1e2},
                        {   'priors_gamma_min': 1e-3,
                            'priors_gamma_max': 1e3,
                            'priors_nu_min'   : 1e-3,
                            'priors_nu_max'   : 1e3   }]

    for ii, prior_set in enumerate(prior_sets):
        if not test:
            GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                        datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                        priors_pulse_start = -.1, priors_pulse_end = 0.6,
                        priors_td_lo = 0,  priors_td_hi = 0.5,
                        directory_label = directory_labels[ii])

            GRB_wrap.offsets = [0, 4000, 8000, -3000]
        else:
            print('Currently running flat priors')
        model_dict = {}
        for key in model_keys:
            model_dict[key] = create_model_from_key(key,
                        custom_name = f'{key}_{directory_labels[ii]}')
        models = [model for key, model in model_dict.items()]
        for mm, model in enumerate(models):
            if not test:
                GRB_wrap._setup_labels(model)
                overwrite_priors = dict()
                for n in range(1, GRB_wrap.num_pulses + 1):
                    for k in ['a', 'b', 'c', 'd']:
                        overwrite_priors[f'gamma_{n}_{k}'] = bilbyUniform(
                        minimum=prior_set['priors_gamma_min'],
                        maximum=prior_set['priors_gamma_max'],
                        latex_label=f'$\\gamma_{n} {k}$', unit=' ')
                        overwrite_priors[f'nu_{n}_{k}'] = bilbyUniform(
                        minimum=prior_set['priors_nu_min'],
                        maximum=prior_set['priors_nu_max'],
                        latex_label=f'$\\nu_{n} {k}$', unit=' ')
                GRB_wrap.overwrite_priors = overwrite_priors

            loop_idx = np.arange(8 * ii + 4 * mm, 8 * ii + 4 * (mm + 1))
            if test:
                print(indices)
                print(ii, mm)
                print('loop_idx = ', loop_idx)
            indx = np.intersect1d(indices, loop_idx) % 4
            if not test:
                GRB_wrap._split_array_job_to_4_channels(models = [model],
                    indices = indx, channels = channels)
            else:
                print('index passed = ', indx)

def run_delta_priors(   indices,
                        model_keys,
                        channels,
                        nSamples,
                        n_per_split,
                        test
                        ):
    directory_label = 'delta_nr'
    if not test:
        GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                    datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                    priors_pulse_start = -.1, priors_pulse_end = 0.6,
                    priors_td_lo = 0,  priors_td_hi = 0.5,
                    directory_label = directory_label)
        GRB_wrap.offsets = [0, 4000, 8000, -3000]
    else:
        print('Currently running delta function priors')
    model_dict = {}
    for key in model_keys:
        model_dict[key] = create_model_from_key(key,
                    custom_name = f'{key}_{directory_label}')
    models = [model for key, model in model_dict.items()]
    for mm, model in enumerate(models):
        if not test:
            GRB_wrap._setup_labels(model)
            overwrite_priors = dict()
            for n in range(1, GRB_wrap.num_pulses + 1):
                for k in ['a', 'b', 'c', 'd']:

                    overwrite_priors[f'gamma_{n}_{k}'] = bilbyDeltaFunction(
                            1,  latex_label = f'$\\gamma$ {n} {k}')

                    overwrite_priors[f'nu_{n}_{k}'] = bilbyDeltaFunction(
                            1,     latex_label = f'$\\nu$ {n} {k}')
            GRB_wrap.overwrite_priors = overwrite_priors

        loop_idx = np.arange(4 *  mm, 4 * (mm + 1))
        if test:
            print(indices)
            print(mm)
            print('loop_idx = ', loop_idx)
        indx = np.intersect1d(indices, loop_idx) % 4
        if not test:
            GRB_wrap._split_array_job_to_4_channels(models = [model],
                indices = indx, channels = channels)
        else:
            print('index passed = ', indx)

def run_gaussian_priors(    indices,
                            model_keys,
                            channels,
                            nSamples,
                            n_per_split,
                            test
                            ):
    directory_label = 'gaussian_nr'
    if not test:
        GRB_wrap = PulseFitter(3770, times = (-.1, 1),
                    datatype = 'tte', nSamples = nSamples, sampler = SAMPLER,
                    priors_pulse_start = -.1, priors_pulse_end = 0.6,
                    priors_td_lo = 0,  priors_td_hi = 0.5,
                    directory_label = directory_label)
        GRB_wrap.offsets = [0, 4000, 8000, -3000]
    else:
        print('Currently running Gaussian priors')
    model_dict = {}
    for key in model_keys:
        model_dict[key] = create_model_from_key(key,
                    custom_name = f'{key}_{directory_label}')
    models = [model for key, model in model_dict.items()]
    for mm, model in enumerate(models):
        if not test:
            GRB_wrap._setup_labels(model)
            overwrite_priors = dict()
            for n in range(1, GRB_wrap.num_pulses + 1):
                overwrite_priors[f'gamma_{n}_a'] = bilbyGaussian(
                    mu = 0.7, sigma = 2.5,  latex_label = f'$\\gamma$ {n} a')
                overwrite_priors[f'gamma_{n}_b'] = bilbyGaussian(
                    mu = 0.3, sigma = 0.4,  latex_label = f'$\\gamma$ {n} b')
                overwrite_priors[f'gamma_{n}_c'] = bilbyGaussian(
                    mu = 0.38, sigma = 0.3, latex_label = f'$\\gamma$ {n} c')
                overwrite_priors[f'gamma_{n}_d'] = bilbyGaussian(
                    mu = 0.5, sigma = 5,    latex_label = f'$\\gamma$ {n} d')
                overwrite_priors[f'nu_{n}_a'] = bilbyGaussian(
                    mu = 2, sigma = 2,      latex_label = f'$\\nu$ {n} a')
                overwrite_priors[f'nu_{n}_b'] = bilbyGaussian(
                    mu = 3.3, sigma = 1.2,  latex_label = f'$\\nu$ {n} b')
                overwrite_priors[f'nu_{n}_c'] = bilbyGaussian(
                    mu = 2.74, sigma = 0.8, latex_label = f'$\\nu$ {n} c')
                overwrite_priors[f'nu_{n}_d'] = bilbyGaussian(
                    mu = 2.7, sigma = 5,    latex_label = f'$\\nu$ {n} d')
            GRB_wrap.overwrite_priors = overwrite_priors

        loop_idx = np.arange(4 * mm, 4 * (mm + 1))
        if test:
            print(indices)
            print(mm)
            print('loop_idx = ', loop_idx)
        indx = np.intersect1d(indices, loop_idx) % 4
        if not test:
            GRB_wrap._split_array_job_to_4_channels(models = [model],
                indices = indx, channels = channels)
        else:
            print('index passed = ', indx)


def analysis_for_3770(indices, test):
    if not test:
        iddy = indices[0]
        time.sleep(iddy * 100)

    nSamples    = 2000
    model_keys  = ['XsL', 'XsXs']
    channels    = [0, 1, 2, 3]
    n_per_split = len(model_keys) * len(channels)

    current_idx = 0
    end_idx     = current_idx + n_per_split * 3 # 3 prior sets for this function

    log_flat_indices = np.intersect1d(indices, np.arange(current_idx, end_idx))
    if len(log_flat_indices) > 0:
        print('Original indices passed to log-flat prior function: ', log_flat_indices)
        run_log_flat_priors(    log_flat_indices, model_keys, channels,
                                nSamples, n_per_split, test)

    current_idx  = end_idx
    end_idx      = current_idx + n_per_split * 3 # 3 prior sets for this function
    flat_indices = np.intersect1d(indices, np.arange(current_idx, end_idx))
    if len(flat_indices) > 0:
        print('Original indices passed to flat prior function: ', flat_indices)
        flat_indices -= current_idx # resets to 0-23 as per OG
        run_flat_priors(    flat_indices, model_keys, channels,
                            nSamples, n_per_split, test)


    current_idx   = end_idx
    end_idx       = current_idx + n_per_split * 1 # 1 prior set for this function
    delta_indices = np.intersect1d(indices, np.arange(current_idx, end_idx))
    if len(delta_indices) > 0:
        print('Original indices passed to delta prior function: ', delta_indices)
        delta_indices -= current_idx # resets to 0-8 as per OG
        run_delta_priors(   delta_indices, model_keys, channels,
                            nSamples, n_per_split, test)

    current_idx   = end_idx
    end_idx       = current_idx + n_per_split * 1 # 1 prior set for this function
    gauss_indices = np.intersect1d(indices, np.arange(current_idx, end_idx))
    if len(gauss_indices) > 0:
        print('Original indices passed to Gaussian prior function: ', gauss_indices)
        gauss_indices -= current_idx # resets to 0-8 as per OG
        run_gaussian_priors(gauss_indices, model_keys, channels,
                            nSamples, n_per_split, test)




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
        SAMPLER = 'nestle'

        # analysis_for_3770(np.arange(64), test = True)
        analysis_for_3770(np.array([6]), test = False) # 63 is max


    else:
        SAMPLER = 'dynesty'
        analysis_for_3770(args.indices, test = False)
