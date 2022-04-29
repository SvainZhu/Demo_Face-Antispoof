from PyQt5 import QtCore, QtGui, QtWidgets, sip
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
import icon_rc
import time
import cv2
import os, itertools
from libsvm.svm import *
from libsvm.svmutil import *
from mat4py import loadmat
from demo_utils.lbp_dct import *
from demo_utils.crop_images import *


def train_classifier_mat(data_dir):

    # 直接读取mat文件
    LBPDCT_train = loadmat(os.path.join(data_dir, 'Comb_LBPDCT_S3_P1_train.mat'))['Comb_train'][0:150]
    data = list(itertools.chain(*LBPDCT_train))

    train_label_vector = loadmat(os.path.join(data_dir, 'label_Comb_train.mat'))['label_Comb_train']
    labels = list(itertools.chain(*train_label_vector))
    print('Preparing trained model')
    prob = svm_problem(labels, data)
    c = 2 ** 15
    g = 2 ** -10
    param = svm_parameter('-t 2 -c %f -b 1 -q -g %f' % (c, g))  # 根据matlab的svm调整参数
    # -t 2（RBF），-c 惩罚因子，-g 调整核函数的gamma参数
    model = svm_train(prob, param)
    return model


def estimate_label(data, labels, model):
    p_label, p_acc, p_val = svm_predict(labels, data, model, '-b 1 -q')
    return p_label


class Ui_MainWindow(object):

    def __init__(self, MainWindow):

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.cap = cv2.VideoCapture()  # 准备获取图像
        self.CAM_NUM = 0

        self.slot_init()  # 设置槽函数
        self.model = train_classifier_mat('LBP_DCT/Comb')

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(765, 645)
        MainWindow.setMinimumSize(QtCore.QSize(765, 645))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/pic/pai.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("华文隶书")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(-1, 50, -1, -1)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_open.setMaximumSize(QtCore.QSize(120, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_open.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/pic/g1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_open.setIcon(icon1)
        self.pushButton_open.setObjectName("pushButton_open")
        self.horizontalLayout.addWidget(self.pushButton_open)
        self.pushButton_take = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_take.sizePolicy().hasHeightForWidth())
        self.pushButton_take.setSizePolicy(sizePolicy)
        self.pushButton_take.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_take.setMaximumSize(QtCore.QSize(100, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_take.setFont(font)
        self.pushButton_take.setIcon(icon)
        self.pushButton_take.setObjectName("pushButton_take")
        self.horizontalLayout.addWidget(self.pushButton_take)
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_close.setMaximumSize(QtCore.QSize(130, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_close.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/pic/down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_close.setIcon(icon2)
        self.pushButton_close.setObjectName("pushButton_close")
        self.horizontalLayout.addWidget(self.pushButton_close)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_face = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_face.sizePolicy().hasHeightForWidth())
        self.label_face.setSizePolicy(sizePolicy)
        self.label_face.setMinimumSize(QtCore.QSize(0, 0))
        self.label_face.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_face.setFont(font)
        self.label_face.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_face.setStyleSheet("background-color: rgb(192, 218, 255);")
        self.label_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_face.setObjectName("label_face")
        self.verticalLayout.addWidget(self.label_face)
        self.verticalLayout.setStretch(2, 5)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Anti-spoofing"))
        # self.label.setText(_translate("MainWindow", "Face Anti-spoofing"))
        self.pushButton_open.setToolTip(_translate("MainWindow", "点击打开摄像头"))
        self.pushButton_open.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_take.setToolTip(_translate("MainWindow", "点击拍照"))
        self.pushButton_take.setText(_translate("MainWindow", "拍照"))
        self.pushButton_close.setToolTip(_translate("MainWindow", "点击关闭摄像头"))
        self.pushButton_close.setText(_translate("MainWindow", "关闭摄像头"))
        self.label_face.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><img src=\":/"
            "newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>"))

    def slot_init(self):
        # 设置槽函数
        self.pushButton_open.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.pushButton_close.clicked.connect(self.closeEvent)
        self.pushButton_take.clicked.connect(self.takePhoto)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                 QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

    def show_camera(self):
        flag, self.image = self.cap.read()

        self.image=cv2.flip(self.image, 1)  # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_face.setScaledContents(True)

    def takePhoto(self):
        if self.timer_camera.isActive():

            test_dir = 'demo\\LBP_DCT'
            class_name = test_dir + '\\photos'
            if not os.path.exists(class_name):
                os.makedirs(class_name)
            face_dir = test_dir + '\\faces'
            if not os.path.exists(face_dir):
                os.makedirs(face_dir)
            for index in range(1, 5):

                cv2.imwrite("%s/%d.jpg" % (class_name, index), self.image)
                print("%s: %d photo" % (class_name, index))
                time.sleep(0.125)

            c = crop_images(class_name, face_dir, 1.0, 64)
            if c == 'False':
                text = 'Can not capture the face !'
                print(text)
                self.label.setText(text)

            else:
                data_test = []
                feature = lbp_dct(face_dir)
                data_test.append(feature)
                labels_test = []
                # model = train_classifier_mat('LBP_DCT\\RE')
                predict_label = estimate_label(data_test, labels_test, self.model)

                if predict_label[0] == 1.0:
                    text = 'The video is true !'
                    print(text)
                    self.label.setText(text)

                else:
                    text = 'The video is fake !'
                    print(text)
                    self.label.setText(text)

    def closeEvent(self):
        if self.timer_camera.isActive():
            ok = QtWidgets.QPushButton()
            cacel = QtWidgets.QPushButton()

            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

            msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cacel.setText(u'取消')

            if msg.exec_() != QtWidgets.QMessageBox.RejectRole:

                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()
                self.label_face.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\""
                                        "/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")


if __name__ == "__main__":

    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
input()
