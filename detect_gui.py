import sys
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class MedicineBagDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("药袋检测系统 - YOLOv8")
        self.resize(1200, 750)  # 稍微增高以容纳文本框

        # 加载模型
        model_path = 'best.pt'
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"无法加载模型:\n{e}")
            sys.exit(1)

        self.conf_threshold = 0.4

        # 初始化状态
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera = False
        self.video_path = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 图像显示区域
        self.label_original = QLabel()
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.set_label_placeholder(self.label_original, "原始图像")

        self.label_detected = QLabel()
        self.label_detected.setAlignment(Qt.AlignCenter)
        self.label_detected.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.set_label_placeholder(self.label_detected, "检测结果")

        self.label_original.setScaledContents(True)
        self.label_detected.setScaledContents(True)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_detected)
        image_layout.setStretchFactor(self.label_original, 1)
        image_layout.setStretchFactor(self.label_detected, 1)

        self.coord_display = QLabel("药袋角点坐标将在此显示...")
        self.coord_display.setWordWrap(True)
        self.coord_display.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.coord_display.setAlignment(Qt.AlignTop)
        self.coord_display.setStyleSheet("border: 1px solid #aaa; padding: 5px;")

        # 按钮
        self.btn_select_image = QPushButton("选择图片")
        self.btn_select_video = QPushButton("选择视频")
        self.btn_start_camera = QPushButton("启动摄像头")
        self.btn_stop = QPushButton("停止")

        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_media)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_select_image)
        button_layout.addWidget(self.btn_select_video)
        button_layout.addWidget(self.btn_start_camera)
        button_layout.addWidget(self.btn_stop)

        # 主布局：图像 + 坐标文本 + 按钮
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.coord_display)  # <<< 新增
        main_layout.addLayout(button_layout)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        central_widget.setLayout(main_layout)

    def set_label_placeholder(self, label, text):
        label.setText(text)
        label.setStyleSheet("color: #999; background-color: #f0f0f0; border: 1px solid #ccc;")

    def display_image(self, label, img):
        if img is None or img.size == 0:
            return
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap)

    def process_frame(self, frame):
        """处理帧，返回原始图、检测图、坐标文本"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        coord_text = ""

        if boxes and len(boxes) > 0:
            for i, box in enumerate(boxes):
                # 获取 xyxy 格式的像素坐标 [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 四个角点（顺时针：左上 → 右上 → 右下 → 左下）
                corners = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2)
                ]
                coord_text += f"药袋 {i+1} 角点坐标:\n"
                coord_text += f"  左上: ({x1}, {y1})\n"
                coord_text += f"  右上: ({x2}, {y1})\n"
                coord_text += f"  右下: ({x2}, {y2})\n"
                coord_text += f"  左下: ({x1}, {y2})\n\n"
        else:
            coord_text = "未检测到药袋"

        return frame.copy(), annotated_frame, coord_text

    def select_image(self):
        self.stop_media()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                orig, det, coords = self.process_frame(img)
                self.display_image(self.label_original, orig)
                self.display_image(self.label_detected, det)
                self.coord_display.setText(coords)
            else:
                QMessageBox.warning(self, "读取失败", "无法读取该图片文件！")

    def select_video(self):
        self.stop_media()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "打开失败", "无法打开视频文件！")
                return
            self.timer.start(30)

    def start_camera(self):
        if self.is_camera:
            return
        self.stop_media()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "摄像头错误", "无法打开摄像头！")
            return
        self.is_camera = True
        self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                orig, det, coords = self.process_frame(frame)
                self.display_image(self.label_original, orig)
                self.display_image(self.label_detected, det)
                self.coord_display.setText(coords)
            else:
                self.stop_media()
                msg = "摄像头已断开" if self.is_camera else "视频播放完毕"
                QMessageBox.information(self, "提示", msg)

    def stop_media(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_camera = False
        self.set_label_placeholder(self.label_original, "原始图像")
        self.set_label_placeholder(self.label_detected, "检测结果")
        self.coord_display.setText("")  # 清空坐标显示


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicineBagDetectionApp()
    window.show()
    sys.exit(app.exec_())