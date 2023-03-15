import time

import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#######################################################
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import subprocess
import requests
import threading
import time
import serial.tools.list_ports
import serial

pp = ""
stat2 = ""
stat = ""
port = ""
hasil = ""

counter = 0
pindah = True

numBreak = 0
numScratch = 0

# sumber = "{}vid2.mkv".format(pp)
sumber = 0

class Detector:
    def __init__(self):
        self.weights=ROOT / 'best3.pt'
        # source=ROOT / 'vid/gabung.mkv'
        # source=0
        self.data=ROOT / 'data/custom_data.yaml'
        self.imgsz=(640, 640)
        self.conf_thres=0.7
        self.iou_thres=0.45
        self.max_det=1000
        # self.device='cpu'
        self.device = 0
        self.view_img=False
        self.save_txt=False
        self.save_conf=True                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        self.save_crop=False
        self.nosave=False
        self.classes=None
        self.agnostic_nms=False
        self.augment=False
        self.visualize=False
        self.update=False
        self.project=ROOT / 'runs/detect'
        self.name='exp'
        self.exist_ok=False
        self.line_thickness=3
        self.hide_labels=False
        self.hide_conf=False
        self.half=False
        self.dnn=False
        self.vid_stride=1

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def set(self, source):
        global stat2
        global jumlah
        global hasil
        global statusKirim, lastStatusKirim
        global numBreak, numScratch, pindah

        batas = 350
        status = False
        #jumlah = "-"
        #hasil = "-"
        statusKirim, lastStatusKirim = False, False

        detek = []

        self.source = str(source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        self.screenshot = self.source.lower().startswith('screen')
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        elif self.screenshot:
            self.dataset = LoadScreenshots(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        self.bs = len(self.dataset)  # batch_size
        # self.bs = 1
        self.vid_path, self.vid_writer = [None] * self.bs, [None] * self.bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        for self.path, self.im, self.im0s, self.vid_cap, self.s in self.dataset:
            with self.dt[0]:
                self.im = torch.from_numpy(self.im).to(self.model.device)
                self.im = self.im.half() if self.model.fp16 else self.im.float()  # uint8 to fp16/32
                self.im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(self.im.shape) == 3:
                    self.im = self.im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                self.visualize = increment_path(self.save_dir / Path(self.path).stem, mkdir=True) if self.visualize else False
                self.pred = self.model(self.im, augment=self.augment, visualize=self.visualize)

            # NMS
            with self.dt[2]:
                self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for self.i, self.det in enumerate(self.pred):  # per image
                self.listPositionX, self.listPositionY = [], []
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    self.p, self.im0, self.frame = self.path[i], self.im0s[i].copy(), self.dataset.count
                    self.imcc = self.im0s[i].copy()
                    self.s += f'{i}: '
                else:
                    self.p, self.im0, self.frame = self.path, self.im0s.copy(), getattr(self.dataset, 'frame', 0)
                    self.imcc = self.im0s.copy()

                self.p = Path(self.p)  # to Path
                self.save_path = str(self.save_dir / self.p.name)  # im.jpg
                self.txt_path = str(self.save_dir / 'labels' / self.p.stem) + ('' if self.dataset.mode == 'image' else f'_{self.frame}')  # im.txt
                self.s += '%gx%g ' % self.im.shape[2:]  # print string
                self.gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                self.imc = self.im0.copy() if self.save_crop else self.im0  # for save_crop
                self.annotator = Annotator(self.im0, line_width=self.line_thickness, example=str(self.names))
                if len(self.det):
                    # Rescale boxes from img_size to im0 size
                    self.det[:, :4] = scale_boxes(self.im.shape[2:], self.det[:, :4], self.im0.shape).round()

                    # Print results
                    for self.c in self.det[:, 5].unique():
                        self.n = (self.det[:, 5] == self.c).sum()  # detections per class
                        self.s += f"{self.n} {self.names[int(self.c)]}{'s' * (self.n > 1)}, "  # add to string

                    # Write results
                    for *self.xyxy, self.conf, cls in reversed(self.det):
                        self.listPositionX.append(self.xyxy[0].cpu().numpy())
                        self.listPositionX.append(self.xyxy[2].cpu().numpy())
                        self.listPositionY.append(self.xyxy[1].cpu().numpy())
                        self.listPositionY.append(self.xyxy[3].cpu().numpy())
                        if self.save_txt:  # Write to file
                            self.xywh = (xyxy2xywh(torch.tensor(self.xyxy).view(1, 4)) / self.gn).view(-1).tolist()  # normalized xywh
                            self.line = (cls, *self.xywh, self.conf) if self.save_conf else (cls, *self.xywh)  # label format
                            with open(f'{self.txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(self.line)).rstrip() % self.line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            self.label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {self.conf:.2f}')
                            self.annotator.box_label(self.xyxy, self.label, color=colors(c, True))
                            detek.append(self.names[c])
                        if self.save_crop:
                            save_one_box(self.xyxy, self.imc, file=self.save_dir / 'crops' / self.names[c] / f'{self.p.stem}.jpg', BGR=True)

                # Stream results
                self.im0 = self.annotator.result()
                cv2.imwrite("simpan2.png", self.im0)
                if self.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)


            # Print time (inference-only)
            self.fps = (self.dt[1].dt)
            print("fps", self.fps)

            if "gear" in detek and len(detek) == 1:
                jumlah = "NORMAL"
                hasil = "OK"
            else:
                hasil = "-"
                jumlah = ""
                if 'break' in detek:
                    jumlah = "{}BREAK ".format(jumlah)
                    hasil = "CACAT"
                if 'scratch' in detek:
                    jumlah = "{}SCRATCH ".format(jumlah)
                    hasil = "CACAT"

            if len(detek) == 0:
                pindah = True

            if pindah:
                numBreak += detek.count("break")
                numScratch += detek.count("scratch")

            statusKirim = True
            cv2.imwrite("{}ini.png".format(pp), self.imcc)

            if len(self.det):
                stat2 = "terdeteksi"
            else:
                stat2 = "tidak terdeteksi"
            # print(stat2)
            print("===========================", jumlah, hasil, detek)
        return self.im0

class StatusDeteksiThread(QThread):
    ubahStatusDeteksi = pyqtSignal()
    def run(self):
        global stat2
        stat2 = "-"
        while True:
            self.ubahStatusDeteksi.emit()
            time.sleep(2)

class StatusObjekThread(QThread):
    ubahStatusObjek = pyqtSignal()
    ubahNumBreak = pyqtSignal()
    ubahNumScratch = pyqtSignal()
    def run(self):
        global jumlah, numScratch, numBreak
        jumlah = "-"
        while True:
            self.ubahStatusObjek.emit()
            self.ubahNumBreak.emit()
            self.ubahNumScratch.emit()
            time.sleep(0.5)

class NumBreakThread(QThread):
    ubahNumBreak = pyqtSignal()
    def run(self):
        global numBreak
        numBreak = "-"
        while True:
            self.ubahNumBreak.emit()
            time.sleep(0.5)

class NumScratchThread(QThread):
    ubahNumScratch = pyqtSignal()
    def run(self):
        global numScratch
        numScratch = "-"
        while True:
            self.ubahNumScratch.emit()
            time.sleep(0.5)

class StatusSerialThread(QThread):
    ubahStatusSerial = pyqtSignal()
    def run(self):
        global hasil, jumlah
        global stat, port
        global ser

        lastHasil = ""
        while True:
            self.myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
            try:
                # print(self.myports)
                if "USB" in self.myports[0][0] or "ACM" in self.myports[0][0]:
                    stat = " Terhubung"
                    print(self.myports[0][0])
                    #try:
                    ser = serial.Serial(str(self.myports[0][0]), 9600)
                    port = str(self.myports[0][0])
                    if hasil == "CACAT":
                        lastHasil = "ki\n\r"
                        ser.write(lastHasil.encode())
                    else:
                        lastHasil = "ka\n\r"
                        ser.write(lastHasil.encode())
                    print(hasil, lastHasil)
                    lastHasil = "-"

                        # if ser.in_waiting > 0:
                        #     try:
                        #         strr = ser.readline()
                        #         data = strr.decode('Ascii')
                        #         print(data)
                        #     except:
                        #         print("ggg")
                    #except:
                    #    print("g")
                else:
                    stat = " Tidak Terhubung"
            except:
                stat = " Tidak Terhubung"
            self.ubahStatusSerial.emit()
            time.sleep(1)
        
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        global stat2
        global jumlah
        global hasil
        global statusServer
        global statusKirim, lastStatusKirim
        global sumber

        status = False
        #hasil = "-"
        statusKirim, lastStatusKirim = False, False

        p1 = Detector()
        cap = cv2.VideoCapture(sumber)

        lebar, tinggi = 550, 350
        status = False
        count = False

        num = 0

        while 1:
            # try:
            ret, img = cap.read()
            if ret:    
                img2 = img.copy()

                lastStatus = status

                height, width = img.shape[:2]
                ym = int((height-tinggi)/2)
                xm = int((width-lebar)/2)

                cv2.imwrite('simpan.png', img2)
                img2 = p1.set('simpan.png')
                
                self.change_pixmap_signal.emit(img2)
            else:
                num += 1
                if num >= 10:
                    cap.release()
                    cap = cv2.VideoCapture(sumber)
                    num = 0
                print("-----------------------------------")
            
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("App")

        sizeObject = QDesktopWidget().screenGeometry(-1)
        # print(sizeObject.width())
        width = int(sizeObject.width()*0.29)

        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        # create a text label
        self.textLabel = QLabel('Webcam')

        ###########################
        self.gbVid = QGroupBox("Camera Stream")

        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        
        self.vb1 = QVBoxLayout()
        self.vb1.addWidget(self.image_label)

        self.gbVid.setLayout(self.vb1)
        ###########################

        #############################################
        self.gbStatSerial = QGroupBox("Indikator Serial")

        self.labelSerial = QLabel()
        self.pixmapSerial = QPixmap('{}iconSerial.png'.format(pp))
        self.pixmapSerial = self.pixmapSerial.scaledToWidth(int(width/7))
        self.labelSerial.setPixmap(self.pixmapSerial)

        self.labelStatusSerial = QLabel("\t-")
        self.labelStatusSerial.setFont(QFont('Arial', int(width/20), QFont.Bold))
        self.labelStatusSerial.setAlignment(Qt.AlignCenter)

        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(QLabel("  "))
        self.hbox2.addWidget(self.labelSerial)
        self.hbox2.addWidget(self.labelStatusSerial)
        self.hbox2.addStretch(1)
        self.gbStatSerial.setLayout(self.hbox2)


        ##############################################
        self.gbStatDeteksi = QGroupBox("Indikator Deteksi")

        self.labelDeteksi = QLabel()
        self.pixmapDeteksi = QPixmap('{}iconDeteksi.png'.format(pp))
        self.pixmapDeteksi = self.pixmapDeteksi.scaledToWidth(int(width/10))
        self.labelDeteksi.setPixmap(self.pixmapDeteksi)

        self.labelStatusDeteksi = QLabel("\t-")
        self.labelStatusDeteksi.setFont(QFont('Arial', int(width/20), QFont.Bold))
        self.labelStatusDeteksi.setAlignment(Qt.AlignCenter)

        self.hbox4 = QHBoxLayout()
        self.hbox4.addWidget(QLabel("  "))
        self.hbox4.addWidget(self.labelDeteksi)
        self.hbox4.addWidget(self.labelStatusDeteksi)
        self.hbox4.addStretch(1)
        self.gbStatDeteksi.setLayout(self.hbox4)

        ##############################################
        self.gbStatObjek = QGroupBox("Indikator Gear")

        self.labelObjek = QLabel()
        self.pixmapObjek = QPixmap('{}iconObjek.png'.format(pp))
        self.pixmapObjek = self.pixmapObjek.scaledToWidth(int(width/10))
        self.labelObjek.setPixmap(self.pixmapObjek)

        self.labelStatusObjek = QLabel("\t-")
        self.labelStatusObjek.setFont(QFont('Arial', int(width/20), QFont.Bold))
        self.labelStatusObjek.setAlignment(Qt.AlignCenter)

        self.hbox5 = QHBoxLayout()
        self.hbox5.addWidget(QLabel("  "))
        self.hbox5.addWidget(self.labelObjek)
        self.hbox5.addWidget(self.labelStatusObjek)
        self.hbox5.addStretch(1)
        self.gbStatObjek.setLayout(self.hbox5)

        ##############################################
        self.gbNumBreak = QGroupBox("Jumlah Break")

        self.labelNumBreak = QLabel("\t-")
        self.labelNumBreak.setFont(QFont('Arial', int(width/20), QFont.Bold))
        self.labelNumBreak.setAlignment(Qt.AlignCenter)

        self.hbox6 = QHBoxLayout()
        self.hbox6.addWidget(QLabel("  "))
        self.hbox6.addWidget(self.labelNumBreak)
        self.hbox6.addStretch(1)
        self.gbNumBreak.setLayout(self.hbox6)

        ##############################################
        self.gbNumScratch = QGroupBox("Jumlah Scratch")

        self.labelNumScratch = QLabel("\t-")
        self.labelNumScratch.setFont(QFont('Arial', int(width/20), QFont.Bold))
        self.labelNumScratch.setAlignment(Qt.AlignCenter)

        self.hbox7 = QHBoxLayout()
        self.hbox7.addWidget(QLabel("  "))
        self.hbox7.addWidget(self.labelNumScratch)
        self.hbox7.addStretch(1)
        self.gbNumScratch.setLayout(self.hbox7)

        ##############################################

        # create a vertical box layout and add the two labels
        self.layout = QGridLayout()
        self.layout.addWidget(self.gbVid, 0, 0, 6, 1)
        self.layout.addWidget(self.gbStatDeteksi, 0, 1,)
        self.layout.addWidget(self.gbStatObjek, 1, 1)
        self.layout.addWidget(self.gbStatSerial, 2, 1)
        self.layout.addWidget(self.gbNumBreak, 3, 1)
        self.layout.addWidget(self.gbNumScratch, 4, 1)
        
        # set the vbox layout as the widgets layout
        self.setLayout(self.layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.thread2 = StatusDeteksiThread()
        self.thread2.ubahStatusDeteksi.connect(self.ubahStatusDeteksi)
        self.thread2.start()

        self.thread3 = StatusObjekThread()
        self.thread3.ubahStatusObjek.connect(self.ubahStatusObjek)
        self.thread3.ubahNumBreak.connect(self.ubahNumBreak)
        self.thread3.ubahNumScratch.connect(self.ubahNumScratch)
        self.thread3.start()

        self.thread5 = StatusSerialThread()
        self.thread5.ubahStatusSerial.connect(self.ubahStatusSerial)
        self.thread5.start()

        #self.thread6 = NumBreakThread()
        #self.thread6.ubahNumBreak.connect(self.ubahNumBreak)
        #self.thread6.start()

        #self.thread7 = NumScratchThread()
        #self.thread7.ubahNumScratch.connect(self.ubahNumScratch)
        #self.thread7.start()

    def ubahNumBreak(self):
        global numBreak
        self.labelNumBreak.setText(" {}".format(numBreak))

    def ubahNumScratch(self):
        global numScratch
        self.labelNumScratch.setText(" {}".format(numScratch))

    def ubahStatusDeteksi(self):
        global stat2
        self.labelStatusDeteksi.setText(" {}".format(stat2.capitalize()))
        
    def ubahStatusObjek(self):
        global jumlah
        self.labelStatusObjek.setText(" {}".format(jumlah))

    def ubahStatusSerial(self):
        global hasil
        global stat, port
        self.labelStatusSerial.setText(str(stat))
      
    # @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    # a.showMaximized()
    sys.exit(app.exec_())
