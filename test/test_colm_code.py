import numpy as np
from bilby.gw.waveform_generator import WaveformGenerator

class MultiWavelet(WaveformGenerator):
    def __init__(
                    self,
                    duration=None,
                    sampling_frequency=None,
                    start_time=0,
                    frequency_domain_source_model=None,
                    time_domain_source_model=None,
                    parameters=None,
                    parameter_conversion=None,
                    waveform_arguments=None,
                    ):
        super(MultiWavelet, self).__init__(
                    duration=duration,
                    sampling_frequency=sampling_frequency,
                    start_time=start_time,
                    frequency_domain_source_model=frequency_domain_source_model,
                    time_domain_source_model=time_domain_source_model,
                    parameters=parameters,
                    parameter_conversion=parameter_conversion,
                    waveform_arguments=waveform_arguments,
                    )

    self._params = list()

    @property
    def n_wavelets(self):
        return len(self._params)

    def _strain_from_model(self, model_data_points, model):
        if self.n_wavelets == 0:
            return dict(plus=np.zeros_like(self.frequency_array, dtype=complex))
        else:
            model_strain = model(model_data_points, **self._params[0])
            for ii in np.arange(1, self.n_wavelets):
                old_strain = model_strain.copy()
                new_strain = model(model_data_points, **self._params[ii])
                for key in new_strain:
                    model_strain[key] = old_strain[key] + new_strain[key]
                    return model_strain
    @classmethod
    def _recursive_add(cls, total_strain, new_strain):
        if isinstance(total_strain, np.ndarray):
            return total_strain + new_strain
        elif isinstance(total_strain, dict):
            for key in total_strain:
                total_strain[key] = cls._recursive_add(
                                    total_strain[key], new_strain[key] )
            return total_strain
    @property
    def parameters(self):
    """ The dictionary of parameters for source model.
    Returns
    -------
    dict: The dictionary of parameter key-value pairs
    """
        return self._parameters
    @parameters.setter
    def parameters(self, parameters):
    """
    Set parameters, this applies the conversion function and then removes
    any parameters which aren't required by the source function.
    """
