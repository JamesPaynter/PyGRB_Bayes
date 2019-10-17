class empty(object):
    """docstring for empty."""

    def __init__(self, arg):
        super(empty, self).__init__()
        self.arg = arg
            
    @staticmethod
    def generate_rates(delta_t, **kwargs):
    # def generate_rates(delta_t, t_0, background, start_1, scale_1, rise_1, decay_1):
        print('***************')
        print(kwargs)
        print('***************')
        for key in kwargs:
            print(key)
        times  = np.cumsum(delta_t)
        times  = np.insert(times, 0, 0.0)
        times += t_0
        widths = np.hstack((delta_t, delta_t[-1]))
        starts = []
        for key in kwargs:
            if 'start' in key:
                starts.append(int(re.sub(r"\D", "", key)))
        if len(starts) > 1:
            num_pulses = np.max(np.array(starts))
        else:
            num_pulses = 1
        print(num_pulses)
        rates = np.zeros(len(times))

        keys  = ['start_', 'scale_', 'rise_', 'decay_', 'times_']
        for i in range(num_pulses):
            keyss = [keys[j] + str(i + 1) for j in range(len(keys))]
            time__      = times - kwargs[keyss[0]]
            time_______ = (time__) * np.heaviside(time__, 0) + 1e-12
            print(time_______)
            print('kwargs[keyss[1]] : ', kwargs[keyss[1]])
            print('kwargs[keyss[2]] : ', kwargs[keyss[2]])
            print('kwargs[keyss[3]] : ', kwargs[keyss[3]])
            rates += kwargs[keyss[1]] * np.exp(
                    - np.power( ( kwargs[keyss[2]] / time_______), 1)
                    - np.power( ( time_______ / kwargs[keyss[3]]), 1))

        rates += kwargs['background']
        print(rates)
        return np.multiply(rates, widths)


    @staticmethod
    def generate_rates1(delta_t, **outer_kwargs):
        ''' Dynamically generate the rate function based on the
            input parameters.

            HOW TO GENERATE A STATIC FUNCTION ONCE, THIS FUNCTION WILL CALL
            ALL THESE EXTRA KEYS ETC EVERY FUNCTION CALL

            IT ONLY NEEDS TO BE DONE ONCE it will be the same after
        '''
        extra_params = {}
        num_pulses   = 0
        starts = []
        for key in outer_kwargs:
            if 'start' in key:
                starts.append(int(re.sub(r"\D", "", key)))
        starts = np.array(starts)
        if len(starts) > 1:
            num_pulses = np.max(starts)
        else:
            num_pulses = 1
        print('The number of pulses is {}'.format(num_pulses))
        list = ['times']
        # for key in extra_params:
        #     print(key)

        # if 'lens' in model:
        #     pass
        # else:
        #     pass
        def rate_function(delta_t, t_0, **inner_kwargs):
            times  = np.cumsum(delta_t)
            times  = np.insert(times, 0, 0.0)
            times += t_0
            widths = np.hstack((delta_t, delta_t[-1]))

            kwargs = inner_kwargs or outer_kwargs
            extra_keys   = []
            extra_params = {}
            for i in range(1, num_pulses + 1):
                extra_keys  += ['{}_{}'.format(list[k],i)
                                for k in range(len(list))]
                extra_params = {k: None for k in extra_keys}
            for i in range(1, num_pulses + 1):
                start_key = 'start_' + str(i)
                times_key = 'times_' + str(i)
                extra_params[times_key] =  ((times - kwargs[start_key]
                ) * np.heaviside(times - kwargs[start_key], 0) + 1e-12 )

            rates = np.zeros(len(times))
            keys  = ['start_', 'scale_', 'rise_', 'decay_', 'times_']
            for i in range(1, num_pulses + 1):
                for j in range(len(keys)):
                    keys[j] += str(i)
                rates += ( kwargs[keys[1]] *
                        np.exp( - (kwargs[keys[2]] / extra_params[keys[4]])
                                - extra_params[keys[4]] / kwargs[keys[3]]) )
            return np.multiply(rates, widths)
        return rate_function
