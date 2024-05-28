# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.6.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(799, 362)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.input = QLabel(self.centralwidget)
        self.input.setObjectName(u"input")
        self.input.setGeometry(QRect(20, 10, 361, 201))
        self.input.setScaledContents(True)
        self.input.setAlignment(Qt.AlignCenter)
        self.output = QLabel(self.centralwidget)
        self.output.setObjectName(u"output")
        self.output.setGeometry(QRect(420, 10, 361, 201))
        self.output.setScaledContents(True)
        self.output.setAlignment(Qt.AlignCenter)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(390, 10, 20, 201))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.detect_image = QPushButton(self.centralwidget)
        self.detect_image.setObjectName(u"detect_image")
        self.detect_image.setGeometry(QRect(10, 220, 371, 41))
        self.exit = QPushButton(self.centralwidget)
        self.exit.setObjectName(u"exit")
        self.exit.setGeometry(QRect(10, 280, 781, 61))
        self.detect_video = QPushButton(self.centralwidget)
        self.detect_video.setObjectName(u"detect_video")
        self.detect_video.setGeometry(QRect(420, 220, 371, 41))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.input.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u539f\u59cb\u6587\u4ef6", None))
        self.output.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u68c0\u6d4b\u7ed3\u679c", None))
        self.detect_image.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u68c0\u6d4b", None))
        self.exit.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa\u7a0b\u5e8f", None))
        self.detect_video.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u68c0\u6d4b", None))
    # retranslateUi

