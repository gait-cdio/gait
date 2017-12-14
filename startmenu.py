import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip,
                             QPushButton, QComboBox, QLabel,
                             QCheckBox, QLineEdit)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication
from collections import namedtuple

returnArgs = namedtuple("returnArgs", "filename cached numOfTrackers method set_thresholds")

class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))

        self.retArg = returnArgs(filename="4farger.mp4",
                                 cached=False,
                                 numOfTrackers=1,
                                 method='marker',
                                 set_thresholds=True)

        self.setToolTip('Brkstr')
        self.setGeometry(500,500,400,400)
        self.setWindowTitle('Welcome human')

        self.menuLbl = QLabel("Type of video",self)
        self.menu = QComboBox(self)
        self.menu.addItem("Marker")
        self.menu.addItem("Markerless")
        self.menu.activated[str].connect(self.onActivated)
        self.menuLbl.move(20,10)
        self.menu.move(20,30)

        self.markerLbl = QLabel("Number of color trackers",self)
        self.markerMenu = QComboBox(self)
        self.markerMenu.addItem("1")
        self.markerMenu.addItem("2")
        self.markerMenu.addItem("3")
        self.markerMenu.addItem("4")
        self.markerMenu.activated[str].connect(self.numOfTrackerChanged)
        self.markerLbl.setVisible(True)
        self.markerLbl.move(20,60)
        self.markerMenu.setVisible(True)
        self.markerMenu.move(20, 80)

        self.videoLbl = QLabel("Filename", self)
        self.videoLbl.move(200,10)
        self.videoName = QLineEdit("4farger.mp4",self)
        self.videoName.textChanged[str].connect(self.videoChanged)
        self.videoName.move(200,30)

        self.cachedLbl = QLabel("Cached",self)
        self.cachedLbl.move(200,60)
        self.cachedBox = QCheckBox(self)
        self.cachedBox.setChecked(False)
        self.cachedBox.stateChanged[int].connect(self.cachedChanged)
        self.cachedBox.move(200,80)

        btn  = QPushButton('EXECUTE', self)
        btn.setToolTip('EXECUTE THE COMMAND')
        btn.clicked.connect(self.returnCallback)
        btn.resize(btn.sizeHint())
        btn.move(20,200)
        self.show()

    def onActivated(self, text):
        if text == "Marker":
            self.markerLbl.setVisible(True)
            self.markerMenu.setVisible(True)
            self.retArg = self.retArg._replace(numOfTrackers = 1, method='marker')
        else:
            self.markerLbl.setVisible(False)
            self.markerMenu.setVisible(False)
            self.retArg = self.retArg._replace(numOfTrackers = 1, method='markerless')

    def numOfTrackerChanged(self, numText):
        self.retArg = self.retArg._replace(numOfTrackers=int(numText))

    def videoChanged(self,text):
        self.retArg = self.retArg._replace(filename = text)

    def cachedChanged(self,num):
        self.retArg = self.retArg._replace(cached = (num == 2))

    def returnCallback(self):
        QCoreApplication.instance().quit()

    def getReturnArgs(self):
        return self.retArg


def start_menu():
    app = QApplication(sys.argv)
    w = WelcomeWindow()
    app.exec_()
    return w.getReturnArgs()



#if __name__ == '__main__':
#    a = start_menu()
#    print(a.filename)
#    print(a.cached)
#    print(a.numOfTrackers)
