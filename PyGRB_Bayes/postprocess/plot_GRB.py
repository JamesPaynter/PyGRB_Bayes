import matplotlib.pyplot as plt


class GRBPlotter(object):
    """docstring for GRBPlotter."""

    def __init__(self, GRB, channels, outdir):
        super(GRBPlotter, self).__init__()
        self.plot_grb(GRB, channels, outdir)

    @staticmethod
    def plot_grb(GRB, channels, outdir):
        """ Plots the GRB given to the plotter class. """
        fig, ax = plt.subplots()
        for i in channels:
            rates = GRB.counts[:,i] / (GRB.bin_right - GRB.bin_left)
            ax.plot(GRB.bin_left, rates,
                    c = GRB.colours[i], drawstyle='steps-mid')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Counts / second')
        plot_name = f'{outdir}/injected_signal'
        fig.savefig(plot_name)
        plt.close()
