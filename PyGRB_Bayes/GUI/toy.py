# # # Filename: toy.py
# # # adapted from https://realpython.com/python-pyqt-gui-calculator/
# #
# # """Simple Hello World example with PyQt5."""
# #
# # import sys
# #
# # # 1. Import `QApplication` and all the required widgets
# # from PyQt5.QtWidgets import QApplication
# # from PyQt5.QtWidgets import QLabel
# # from PyQt5.QtWidgets import QWidget
# # from PyQt5.QtWidgets import QLineEdit
# # from PyQt5.QtWidgets import QHBoxLayout
# # from PyQt5.QtWidgets import QPushButton
# #
# # from PyQt5.QtWidgets import QGridLayout
# #
# #
# #
# # # 2. Create an instance of QApplication
# # app = QApplication(sys.argv)
# #
# #
# #
# # # 3. Create an instance of your application's GUI
# # # window is an instance of QWidget
# # window = QWidget()
# # window.setWindowTitle('Title')
# # # The first two parameters are the x and y coordinates at which the window
# # # will be placed on the screen. The third and fourth parameters
# # # are the width and height of the window.
# # # window.setGeometry(0, 0, 500, 300)
# # # window.move(60, 15)
# # # # helloMsg = QLabel('<h2>Hello World!</h2>', parent=window)
# # # # helloMsg.move(60, 100)
# # # layout = QHBoxLayout()
# # # layout.addWidget(QPushButton('Left'))
# # # layout.addWidget(QPushButton('Center'))
# # # layout.addWidget(QPushButton('Right'))
# # # window.setLayout(layout)
# #
# # layout = QGridLayout()
# # layout.addWidget(QPushButton('Button (0, 0)'), 0, 0, 1, 3)
# # # layout.addWidget(QPushButton('Button (0, 1)'), 0, 1)
# # # layout.addWidget(QPushButton('Button (0, 2)'), 0, 2)
# # # layout.addWidget(QPushButton('Button (1, 0)'), 1, 0)
# # # layout.addWidget(QPushButton('Button (1, 1)'), 1, 1)
# # # layout.addWidget(QPushButton('Button (1, 2)'), 1, 2)
# # layout.addWidget(QPushButton('Button (2, 0)'), 2, 0)
# # layout.addWidget(QPushButton('Button (2, 0)'), 2, 1)
# # layout.addWidget(QPushButton('Button (2, 0)'), 2, 2)
# # # layout.addWidget(QPushButton('Button (2, 1) + 2 Columns Span'), 2, 1, 1, 2)
# # window.setLayout(layout)
# #
# # # 4. Show your application's GUI
# # window.show()
# #
# # # 5. Run your application's event loop (or main loop)
# # sys.exit(app.exec_())
#
# # Filename: main_window.py
#
# """Main Window-Style application."""
#
# import sys
#
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtWidgets import QLabel
# from PyQt5.QtWidgets import QMainWindow
# from PyQt5.QtWidgets import QStatusBar
# from PyQt5.QtWidgets import QToolBar
#
# class Window(QMainWindow):
#     """Main Window."""
#     def __init__(self, parent=None):
#         """Initializer."""
#         super().__init__(parent)
#         self.setWindowTitle('QMainWindow')
#         self.setCentralWidget(QLabel("I'm the Central Widget"))
#         self._createMenu()
#         self._createToolBar()
#         self._createStatusBar()
#
#     def _createMenu(self):
#         self.menu = self.menuBar().addMenu("&Menu")
#         self.menu.addAction('Exit', self.close)
#         self.menu.addAction('Hello')
#
#     def _createToolBar(self):
#         tools = QToolBar()
#         self.addToolBar(tools)
#         tools.addAction('A', self.close)
#         tools.addAction('B', self.close)
#         tools.addAction('C', self.close)
#         tools.addAction('D', self.close)
#
#     def _createStatusBar(self):
#         status = QStatusBar()
#         status.showMessage("I'm the Status Bar")
#         self.setStatusBar(status)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = Window()
#     win.show()
#     sys.exit(app.exec_())

#
# import tkinter as tk
# import random
#
# class ServoDrive(object):
#     # simulate values
#     def getVelocity(self): return random.randint(0,50)
#     def getTorque(self): return random.randint(50,100)
#
# class Example(tk.Frame):
#     def __init__(self, *args, **kwargs):
#         tk.Frame.__init__(self, *args, **kwargs)
#         self.servo = ServoDrive()
#         self.canvas = tk.Canvas(self, background="black")
#         self.canvas.pack(side="top", fill="both", expand=True)
#
#         # create lines for velocity and torque
#         self.velocity_line = self.canvas.create_line(0,0,0,0, fill="red")
#         self.torque_line = self.canvas.create_line(0,0,0,0, fill="blue")
#
#         # start the update process
#         self.update_plot()
#
#     def update_plot(self):
#         v = self.servo.getVelocity()
#         t = self.servo.getTorque()
#         self.add_point(self.velocity_line, v)
#         self.add_point(self.torque_line, t)
#         self.canvas.xview_moveto(1.0)
#         self.after(100, self.update_plot)
#
#     def add_point(self, line, y):
#         coords = self.canvas.coords(line)
#         x = coords[-2] + 1
#         coords.append(x)
#         coords.append(y)
#         coords = coords[-200:] # keep # of points to a manageable size
#         self.canvas.coords(line, *coords)
#         self.canvas.configure(scrollregion=self.canvas.bbox("all"))
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     Example(root).pack(side="top", fill="both", expand=True)
#     root.mainloop()




import sys
import os
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import functools
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading



def setCustomSize(x, width, height):
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

        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")

        # Create FRAME_A
        self.FRAME_A = QtWidgets.QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210,210,235,255).name())
        self.LAYOUT_A = QtWidgets.QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)

        # Place the zoom button
        self.zoomBtn = QtWidgets.QPushButton(text = 'zoom')
        setCustomSize(self.zoomBtn, 100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0,0))

        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))

        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()

        self.show()

    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)


class CustomFigCanvas(FigureCanvas, TimedAnimation):

    def __init__(self):

        self.addedData = []
        print(matplotlib.__version__)

        # The data
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50

        # The window
        self.fig = Figure(figsize=(5,5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)


        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0, 100)


        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])

    def addData(self, value):
        self.addedData.append(value)

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()


    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])


        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(float)



def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    # Simulate some data
    n = np.linspace(0, 499, 500)
    y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
    i = 0

    while(True):
        if(i > 499):
            i = 0
        time.sleep(0.1)
        mySrc.data_signal.emit(y[i]) # <- Here you emit a signal!
        i += 1
    ###
###



if __name__== '__main__':
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()


    sys.exit(app.exec_())
