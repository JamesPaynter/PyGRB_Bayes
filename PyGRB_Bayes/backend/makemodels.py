def create_model_dict(  lens, count_FRED, count_FREDx, count_sg, count_bes,
                        **kwargs):
    model = {}
    model['lens']        = lens
    model['count_FRED']  = count_FRED
    model['count_FREDx'] = count_FREDx
    model['count_sg']    = count_sg
    model['count_bes']   = count_bes
    if kwargs:
        for kwarg in kwargs:
            model[kwarg] = kwargs[kwarg]
    return model

def make_singular_models():
    ''' Create the full array of 1-pulse models. '''
    # TODO create gaussian
    models       = {}
    models['F']  = create_model_dict(   lens = False, count_FRED  = [1],
                                        count_FREDx = [],
                                        count_sg    = [],
                                        count_bes   = [],
                                        name = 'FRED')
    models['Fs'] = create_model_dict(   lens = False, count_FRED  = [1],
                                        count_FREDx = [],
                                        count_sg    = [1],
                                        count_bes   = [],
                                        name = 'FRED sg residual')
    models['Fb'] = create_model_dict(   lens = False, count_FRED  = [1],
                                        count_FREDx = [],
                                        count_sg    = [],
                                        count_bes   = [1],
                                        name = 'FRED bes residual')
    models['X']  = create_model_dict(   lens = False, count_FRED  = [],
                                        count_FREDx = [1],
                                        count_sg    = [],
                                        count_bes   = [],
                                        name = 'FREDx')
    models['Xs'] = create_model_dict(   lens = False, count_FRED  = [],
                                        count_FREDx = [1],
                                        count_sg    = [1],
                                        count_bes   = [],
                                        name = 'FREDx sg residual')
    models['Xb'] = create_model_dict(   lens = False, count_FRED  = [],
                                        count_FREDx = [1],
                                        count_sg    = [],
                                        count_bes   = [1],
                                        name = 'FREDx bes residual')
    return models

def _get_pos_from_key(key, char):
    """ Returns a list of the indices where char appears in the string.
        Pass in a list of only the pulses no residuals (ie capital letters)
        +1 is because the pulses are index from 1.
        """
    return [i+1 for i, c in enumerate(key) if c == char]

def _get_pos_from_idx(key, idx_array, char):
    return [idx_array[i] for i, c in enumerate(key) if c == char]

def create_model_from_key(key):
    assert(isinstance(key, str))
    kwargs = {}
    kwargs['lens'] = True if 'L' in key else False
    key = key.strip('L')
    # Gaussian, FRED, FREDx, Convolution
    # TODO allow SG or BES to be standalone pulses
    # as needed, may add bugs down the lineif implemented naively
    pulse_types = ['G', 'F', 'X', 'C']#, 'S', 'B']
    pulse_kwargs= ['count_Gauss', 'count_FRED', 'count_FREDx', 'count_conv']
                    # 'count_sg', 'count_bes']
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
    return model
