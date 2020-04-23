class AbstractBasePlot(object):
    """ Absracts settings for plots for presentations, papers, etc. """

    def __init__(self, plot_type):
        super(AbstractBasePlot, self).__init__()

        self.plot_dict = dict()

        if p_type == 'presentation':
            # specific to single_channel_plot
            self.plot_dict['heights']   = [3, 1] + [0.6]
            # specific to single_channel_plot
            n_axes  = 3
            print('WIDTH CHANGED FROM 3.321 TO 8 FOR PRESENTATIONS')
            self.plot_dict['width']     = 8
            self.plot_dict['p_type']    = 'presentation'
            self.plot_dict['linewidth'] = 1.0
            self.plot_dict['font_size'] = 22
            self.plot_dict['ext']       = 'png'

        elif p_type == 'paper_one_col':
            # specific to single_channel_plot
            self.plot_dict['heights']   = [3, 1, 3, 3] + [0.6]
            self.plot_dict['width']     = 3.54
            # self.plot_dict['width']     = 3.321
            self.plot_dict['p_type']    = 'paper_one_col'
            self.plot_dict['linewidth'] = 0.4
            self.plot_dict['font_size'] = 8
            self.plot_dict['ext']       = 'pdf'


        elif p_type == 'paper_two_col':
            # specific to single_channel_plot
            self.plot_dict['heights']   = [3, 1, 3, 3] + [0.6]
            self.plot_dict['width']     = 7.25
            self.plot_dict['p_type']    = 'paper_two_col'
            self.plot_dict['linewidth'] = 0.4
            self.plot_dict['font_size'] = 8
            self.plot_dict['ext']       = 'pdf'


        elif p_type == 'thesis':
            print("Not yet implemented, try 'paper_two_col'.")
            self.plot_dict['ext']       = 'pdf'


        elif p_type == 'animation':
            print("Not yet implemented, try 'presentation'.")
            self.plot_dict['ext']       = 'png'


        else:
            print('Please specify the purpose of this plot')
