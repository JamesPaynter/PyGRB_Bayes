import numpy as np
import pandas as pd
from abc import ABCMeta

import bilby
from prettytable import PrettyTable


from PyGRB.backend.makemodels import create_model_from_key
from PyGRB.backend.makemodels import make_one_pulse_models

class EvidenceTables(object):
    """
    Defines the :class:`~Admin` class of the *PyGRB* package.
    This is an abstract class that contains the private methods of the
    :class:`~BilbyObject` class. These methods predominantly translate fitting
    parameters into labels for file or folder names and vice versa.
    """

    def __init__(self):
        super(EvidenceTables, self).__init__()

    def get_evidence_table(self,    models, channels,
                                    keys = None, return_tex = True):
        """
        A function to return the evidence tables for the given channels
        for the specified models. Returns one table per channel.
        Parameters
        ----------
        models : list.
            A list of models to be evaluated, and model evidence tabulated.
            Each model will be a dict {}.
        channels : list.
            A list of integers
        keys : list.
            A list of model names in the form of shorthand model keys. To be
            included if the model dicts in :param:`~models` are unnamed.
        return_tex : bool.
            If *True* returns the evidence tables as separate .tex tables for
            each channel. If *False* returns the evidence tables as a single
            .txt file. All files are saved in the relevant base directory.
        """
        if return_tex:
            self._evidence_table_to_latex(models, channels, keys)
        else:
            self._evidence_table_to_txt(models, channels, keys)

    def _evidence_table_to_txt(self, models, channels, keys):
        """
        Returns the evidence tables as a single .txt file.
        See :meth:`~get_evidence_table`.
        """
        self.tlabel = self._get_trigger_label()
        self._get_base_directory()
        directory = self.base_folder
        Z_file = f'{directory}/evidence_table_T{self.trigger}_nlive{self.nSamples}.txt'
        open(Z_file, 'w').close()
        for i in channels:
            x = PrettyTable(['Model', 'ln Z', 'error'])
            x.align['Model'] = "l" # Left align models
            # One space between column edges and contents (default)
            x.padding_width = 1
            for k in range(len(models)):
                if 'name' not in [*models[k]]:
                    models[k]['name'] = keys[k]
                self._setup_labels(models[k])
                result_label = f'{self.fstring}_result_{self.clabels[i]}'
                open_result  = f'{self.outdir}/{result_label}_result.json'
                try:
                    result = bilby.result.read_in_result(filename=open_result)
                    x.add_row([ models[k]['name'],
                                f'{result.log_evidence:.2f}',
                                f'{result.log_evidence_err:.2f}'])
                except:
                    print(f'Could not find {open_result}')

            min_e = np.inf
            for row in x:
                row.border = False
                row.header = False
                e = float(row.get_string(fields=['ln Z']).strip())
                if e < min_e:
                    min_e = e
            bayes_facs = []
            for row in x:
                row.border = False
                row.header = False
                e = float(row.get_string(fields=['ln Z']).strip())
                bayes_facs.append(f'{e - min_e:.2f}')
            x.add_column('ln BF', bayes_facs)
            # indentation should be same as k loop
            with open(Z_file, 'a') as w:
                w.write(f'Channel {i+1}')
                w.write(str(x))
                w.write('')

    def _evidence_table_to_latex(self, models, channels, keys):
        """
        Returns the evidence tables as a separate .tex files for each channel.
        See :meth:`~get_evidence_table`.
        """
        self.tlabel = self._get_trigger_label()
        self._get_base_directory()
        directory = self.base_folder
        columns = ['Model', 'ln evidence', 'ln error', 'ln BF']
        index = np.arange(len(models))
        for i in channels:
            channel_df = pd.DataFrame(columns=columns, index = index)
            for k in range(len(models)):
                if 'name' not in [*models[k]]:
                    models[k]['name'] = keys[k]
                self._setup_labels(models[k])
                result_label = f'{self.fstring}_result_{self.clabels[i]}'
                open_result  = f'{self.outdir}/{result_label}_result.json'
                try:
                    result = bilby.result.read_in_result(filename=open_result)
                    new_row = { 'Model' : [models[k]['name']],
                                'ln evidence' : [result.log_evidence],
                                'ln error' : [result.log_evidence_err],
                                'ln BF' : [result.log_evidence]
                              }
                except:
                    print(f'Could not find {open_result}')
                    new_row = { 'Model' : [models[k]['name']]}
                df = pd.DataFrame(new_row, index = [k])
                channel_df.update(df)
            base_BF = channel_df['ln evidence'].min()
            for k in range(len(models)):
                channel_df.loc[[k], ['ln BF']] = channel_df.loc[[k], ['ln BF']] - base_BF
            print(channel_df.to_latex(  index=False, float_format="{:0.2f}".format))
            channel_df.to_latex(f'{directory}/BF_table_ch_{i+1}.tex',
                                index=False, float_format="{:0.2f}".format)


    def get_evidence_singular(self):
        """
        A method to generate the single pulse models and evaluate the evidence
        for each model.
        """
        self.models = make_one_pulse_models()
        keys = self.models.keys()
        models = [model for key, model in self.models.items()]
        self.get_evidence_table(models = models, return_tex = True,
                                channels = [0, 1, 2, 3], keys = keys)

    def get_evidence_singular_lens(self):
        """
        A method to evaluate the possible two pulse models, including one-pulse
        lens models.
        """
        lens_keys = ['FL', 'FsL', 'XL', 'XsL']
        fred_keys = ['FF', 'FsF', 'FFs', 'FsFs']
        frex_keys = ['XX', 'XsX', 'XXs', 'XsXs']
        mixx_keys = ['FX', 'XF', 'FsX', 'XsF', 'FXs', 'XFs']
        keys = lens_keys + fred_keys + frex_keys + mixx_keys

        # keys = ['FF', 'FL', 'FsFs', 'FsL', 'XX', 'XL', 'XsXs', 'XsL']
        # keys+= ['FsF', 'FFs', 'XsX', 'XXs', 'FsX', 'XsF', 'FXs', 'XFs']
        # keys+= ['FbFb', 'FbL', 'XbXb', 'XbL']
        # keys+= ['FbF', 'FFb', 'XbX', 'XXb', 'FbX', 'XbF', 'FXb', 'XFb']
        self.models = {}
        for key in keys:
            self.models[key] = create_model_from_key(key)
        models = [model for key, model in self.models.items()]
        self.get_evidence_table(models = models, return_tex = False,
                                channels = [0, 1, 2, 3], keys = keys)
        self.get_evidence_table(models = models, return_tex = True,
                                channels = [0, 1, 2, 3], keys = keys)

    def get_evidence_from_models(self, model_dict):
        """
        A method to generate the single pulse models and evaluate the evidence
        for each model.
        """
        keys = model_dict.keys()
        models = [model for key, model in model_dict.items()]
        self.get_evidence_table(models = models, return_tex = False,
                                channels = [0, 1, 2, 3], keys = keys)
        self.get_evidence_table(models = models, return_tex = True,
                                channels = [0, 1, 2, 3], keys = keys)
