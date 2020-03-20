import copy

def create_model_dict(lens = False, **kwargs):
    model = {}
    model['lens'] = lens
    for kwarg in kwargs:
        model[kwarg] = kwargs[kwarg]
    return model

def make_one_pulse_models():
    keys = ['G',  'F',  'X',  'C' ]
    keys+= ['Gs', 'Fs', 'Xs', 'Cs']
    keys+= ['Gb', 'Fb', 'Xb', 'Cb']
    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key)
    return model_dict

def make_two_pulse_models():
    # 18 X 4 = 72 arrays to the job
    lens_keys = ['FL', 'FsL', 'XL', 'XsL']
    fred_keys = ['FF', 'FsF', 'FFs', 'FsFs']
    frex_keys = ['XX', 'XsX', 'XXs', 'XsXs']
    mixx_keys = ['FX', 'XF', 'FsX', 'XsF', 'FXs', 'XFs']
    keys = lens_keys + fred_keys + frex_keys + mixx_keys

    model_dict = {}
    for key in keys:
        model_dict[key] = create_model_from_key(key)
    return model_dict

def _get_pos_from_key(key, char):
    """ Returns a list of the indices where char appears in the string.
        Pass in a list of only the pulses no residuals (ie capital letters)
        +1 is because the pulses are index from 1.
        """
    return [i+1 for i, c in enumerate(key) if c == char]

def _get_pos_from_idx(key, idx_array, char):
    return [idx_array[i] for i, c in enumerate(key) if c == char]

def create_model_from_key(key, custom_name = None):
    assert(isinstance(key, str))
    kwargs = {}
    kwargs['lens'] = True if 'L' in key else False
    if custom_name:
        name = custom_name
    else:
        name = copy.deepcopy(key)
    key = key.strip('L')
    # Gaussian, FRED, FREDx, Convolution
    pulse_types = ['G', 'F', 'X', 'C']
    pulse_kwargs= ['count_gauss', 'count_FRED', 'count_FREDx', 'count_conv']
    res_types   = ['s', 'b']
    res_kwargs  = ['count_sg', 'count_bes']
    # list of capital letters only (ie pulses)
    pulse_keys  = ''.join([c for c in key if c.isupper()])
    pulse_list  = []
    res_list    = []
    for i, char in enumerate(pulse_types):
        # list of indices where current char ('G', 'F' etc.) appears
        idx_list = _get_pos_from_key(pulse_keys, char)
        # appends this to list of pulses
        pulse_list += idx_list
        # also adds this list to the kwargs dict to be passed to the model
        kwargs[pulse_kwargs[i]] = idx_list
    # sort the list of pulses
    pulse_list.sort()
    # indices of where pulses appear in original string
    pulse_indices = [i for i, c in enumerate(key) if c.isupper()]
    idx_array = [0 for i in range(len(key))]
    # idx_array[i] = pulse_indices[]
    for i in range(len(key)):
        idx_array[i] = 0

    for i, (j, k) in enumerate(zip(pulse_list, pulse_indices)):
        idx_array[k] = j
    for i in range(1, len(key)):
        if idx_array[i] == 0:
            idx_array[i] = idx_array[i-1]

    for i, char in enumerate(res_types):
        kwargs[res_kwargs[i]] = _get_pos_from_idx(key, idx_array, char)

    model = create_model_dict(**kwargs)
    if model.get('name') is None:
        model['name'] = name
    return model


if __name__ == '__main__':
    pass
