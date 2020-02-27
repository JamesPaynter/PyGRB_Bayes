import numpy as np

class MakeKeys(object):
    '''
        Doc string goes here.
    '''

    def __init__(self,  count_FRED, count_FREDx, count_sg, count_bes,
                        lens, **kwargs):
        super(MakeKeys, self).__init__()

        # self.test_pulse_keys(count_FRED, count_FREDx)
        # self.test_residual_keys(count_sg, count_bes)

        self.count_gauss  = [] #count_gauss
        self.count_FRED   = count_FRED
        self.count_FREDx  = count_FREDx
        self.count_convo  = [] #count_convo
        self.count_sg     = count_sg
        self.count_bes    = count_bes

        self.pulse_counts = [self.count_gauss, self.count_FRED,
                             self.count_FREDx, self.count_convo]
        self.res_counts   = [self.count_sg, self.count_bes]

        self.rate_counts  = self.pulse_counts + self.res_counts

        self.gauss_list   = ['start', 'scale', 'sigma']
        self.FRED_list    = ['start', 'scale', 'tau', 'xi']
        self.FREDx_list   = self.FRED_list.copy() + ['gamma', 'nu']

        self.convo_list   = ['start', 'scale', 'other', 'parameters']

        self.res_sg_list  = ['sg_A', 'res_begin', 'sg_lambda', 'sg_omega', 'sg_phi']
        self.res_bes_list = [   'bes_A', 'bes_Omega', 'bes_s',
                                'res_begin', 'bes_Delta']

        self.rate_lists   = [self.gauss_list, self.FRED_list, self.FREDx_list,
                           self.convo_list, self.res_sg_list, self.res_bes_list]

        self.lens         = lens
        self.lens_list    = ['time_delay', 'magnification_ratio']

        self.keys = []
        self.get_max_pulse()
        self.get_residual_list()
        self.fill_keys_list()

    def fill_list(self, key_list, array):
        return ['{}_{}'.format(key_list[k], i)  for k in range(len(key_list))
                                                for i in array]

    def fill_keys_list(self):
        if self.lens:
            self.keys += self.lens_list
        self.keys += ['background']
        for p_list, p_type in zip(self.rate_lists, self.rate_counts):
            self.keys += self.fill_list(p_list,  p_type)

        # self.keys += self.fill_list(self.FRED_list,  self.count_FRED)
        # self.keys += self.fill_list(self.FREDx_list, self.count_FREDx)
        # self.keys += self.fill_list(self.res_sg_list,  self.count_sg)
        # self.keys += self.fill_list(self.res_bes_list, self.count_bes)

    def get_max_pulse(self):
        mylist = self.count_FRED + self.count_FREDx
        ## set gets the unique values of the list
        myset  = set(mylist)
        try:
            self.max_pulse = max(myset) ## WILL NEED EXPANDING
        except:
            self.max_pulse = 0

    def get_residual_list(self):
        mylist = self.count_sg + self.count_bes
        myarr  = np.array(mylist)
        myarr  = np.unique(myarr)
        mysort = np.sort(myarr)
        mylist = [mysort[i] for i in range(len(mysort))]
        self.residual_list = mylist

    # @staticmethod
    # def test_pulse_keys(count_FRED, count_FREDx):
    #     self.test_continuous(count_FRED, count_FREDx)
    #     self.test_no_duplicates(count_FRED, count_FREDx)
    #
    # @staticmethod
    # def test_residual_keys(count_sg, count_bes):
    #     self.test_no_duplicates(count_sg, count_bes)
    #
    # @staticmethod
    # def test_no_duplicates(*args):
    #     pass
    #
    # @staticmethod
    # def test_continuous(*args):
    #     s = set([arg for arg in *args])
    #     n = max(s)
    #     pulse_list = [i+1 for i in range(n)]

if __name__ == '__main__':
    pass
