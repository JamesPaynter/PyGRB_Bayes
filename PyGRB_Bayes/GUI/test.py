import sys
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


import threading


def setCustomSize(x, width, height):
    """ copied verbatim """
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(x.sizePolicy().hasHeightForWidth())
    x.setSizePolicy(sizePolicy)
    x.setMinimumSize(QtCore.QSize(width, height))
    x.setMaximumSize(QtCore.QSize(width, height))


class CustomMainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window (= the WHOLE GUI)
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")


        # Create FRAME_A
        self.FRAME_A = QtWidgets.QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }"
                                    % QtGui.QColor(210,210,235,255).name())
        self.LAYOUT_A = QtWidgets.QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)

        # Place the trigger button
        self.TriggerButton = QtWidgets.QLineEdit(text = 'Trigger')
        setCustomSize(self.TriggerButton, 100, 50)
        self.TriggerButton.returnPressed.connect(self.TriggerButtonAction)
        self.LAYOUT_A.addWidget(self.TriggerButton, *(0,0))

        # Place the trigger button
        self.DataButton = QtWidgets.QLineEdit(text = 'datatype')
        setCustomSize(self.DataButton, 100, 50)
        self.DataButton.returnPressed.connect(self.DataButtonAction)
        self.LAYOUT_A.addWidget(self.DataButton, *(1,0))


        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1,2,1))


        self.show()


    def TriggerButtonAction(self):
        self.myFig.getTrigger(int(self.TriggerButton.text()))

    def DataButtonAction(self):
        print(self.DataButton.text())


from PyGRB_Bayes.main.fitpulse import PulseFitter
import matplotlib.pyplot as plt
class CustomFigCanvas(FigureCanvas):

    def __init__(self):
        self.fig  = Figure(figsize=(5,5), dpi=100)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        # TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)

        self.datatype = 'discsc'
        self.times = 'T90'

    def getTrigger(self, trigger):
        self.GRB = PulseFitter(trigger, times = self.times,
                    datatype = self.datatype, nSamples = None, sampler = None,
                    priors_pulse_start = None, priors_pulse_end = None)

        axes = self.GRB.plot_naked(  channels = [0, 1, 2, 3])
        self.axes = axes
        # plt.show(fig)
        self.draw()
        plt.autoscale()




if __name__== '__main__':
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()


    sys.exit(app.exec_())
