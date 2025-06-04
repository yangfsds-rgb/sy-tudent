import sys
import cv2
import qdarkstyle
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QGroupBox, QTextEdit)

from focus_analyzer_界面调用副本 import FocusAnalyzer  # 假设您的分析类在此文件中

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    analysis_result_signal = pyqtSignal(dict)

    def __init__(self, analyzer, source=0):
        super().__init__()
        self.analyzer = analyzer
        self.source = source
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                processed_frame, scores = self.analyzer.process_frame(frame)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
                self.analysis_result_signal.emit(scores)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class FocusAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.video_thread = None
        self.init_ui()
        self.init_connections()

    def init_ui(self):
        self.setWindowTitle("课堂专注度分析系统")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout()

        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_image = QPushButton("加载图片")
        self.btn_load_video = QPushButton("加载视频")
        self.btn_camera = QPushButton("启动摄像头")
        self.btn_analyze = QPushButton("实时分析")
        self.btn_export = QPushButton("导出结果")

        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(self.btn_load_video)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_analyze)
        control_layout.addWidget(self.btn_export)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)

        # 右侧显示区域
        display_panel = QGroupBox("分析结果")
        display_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.result_text)
        display_panel.setLayout(display_layout)

        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_panel, 3)

        self.setCentralWidget(main_widget)
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.result_text.setStyleSheet("font: 12pt 'Consolas';")

    def init_connections(self):
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_analyze.clicked.connect(self.toggle_analysis)
        self.btn_export.clicked.connect(self.export_results)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if path:
            try:
                self.analyzer = FocusAnalyzer(path)
                QMessageBox.information(self, "成功", "模型加载成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败：{str(e)}")

    def load_image(self):
        if not self.check_analyzer():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)"
        )
        if path:
            frame = cv2.imread(path)
            processed_frame, scores = self.analyzer.process_frame(frame)
            self.display_frame(processed_frame)
            self.display_scores(scores)

    def load_video(self):
        if not self.check_analyzer():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi)"
        )
        if path:
            self.start_video_analysis(path)

    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.stop_video_analysis()
            self.btn_camera.setText("启动摄像头")
        else:
            self.start_video_analysis(0)
            self.btn_camera.setText("停止摄像头")

    def toggle_analysis(self):
        if self.video_thread and self.video_thread.isRunning():
            self.stop_video_analysis()
            self.btn_analyze.setText("开始分析")
        else:
            self.start_video_analysis(self.current_source)
            self.btn_analyze.setText("停止分析")

    def start_video_analysis(self, source):
        if self.analyzer is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        self.stop_video_analysis()
        self.video_thread = VideoThread(self.analyzer, source)
        self.video_thread.change_pixmap_signal.connect(self.display_frame)
        self.video_thread.analysis_result_signal.connect(self.display_scores)
        self.video_thread.start()

    def stop_video_analysis(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

    def display_frame(self, image):
        if isinstance(image, QImage):
            pixmap = QPixmap.fromImage(image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def display_scores(self, scores):
        text = "实时分析结果：\n"
        for behavior, score in scores.items():
            text += f"{behavior}: {score:.2f}\n"
        self.result_text.setText(text)

    def export_results(self):
        # 实现结果导出逻辑
        pass

    def check_analyzer(self):
        if self.analyzer is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return False
        return True

    def closeEvent(self, event):
        self.stop_video_analysis()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FocusAnalyzerApp()
    window.show()
    sys.exit(app.exec_())