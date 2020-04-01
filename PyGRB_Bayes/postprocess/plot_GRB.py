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


#
#
# #SBATCH --array=1-80
# #SBATCH --time=300:00:00
#
# index=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79)
