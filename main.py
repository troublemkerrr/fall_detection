import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QUrl, QVariant
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
                            QFileDialog, QTableWidget, QTableWidgetItem,
                            QComboBox, QLineEdit, QDialog, QFormLayout)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sqlite3
from minio import Minio
from minio.error import S3Error
import os
import uuid
import tempfile
from pipline import FallDetectionPipeline
from database import FallDetectionDB
from fall.miniodb import VideoStorage
from config import Config
from collections import deque


class VideoCaptureThread(QThread):
    """视频采集与处理线程"""
    frame_processed = pyqtSignal(np.ndarray)  # 处理后的帧信号
    fall_detected = pyqtSignal(str, list)     # (event_id, frames)信号
    
    def __init__(self, camera_source=0, config=None, parent=None):
        super().__init__(parent)
        self.camera_source = camera_source
        self.config = config
        self.running = False
        self.recording = False
        self.frames_buffer = []
        self.current_event_id = None
        self.pre_fall_status = False
        self.frame_queue = deque(maxlen=5)
        
        # 初始化处理管道
        self.pipeline = FallDetectionPipeline(config)
        
    def run(self):
        """主采集循环"""
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
            
        self.running = True
        frame_count = 0
        target_fps = self.config.TARGET_FPS if hasattr(self.config, 'TARGET_FPS') else 8
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # 按目标帧率处理
            if frame_count % (30 // target_fps) == 0:
                processed_frame, _ = self.process_frame(frame, frame_count)
                self.frame_processed.emit(processed_frame)
                
                # 录制逻辑
                if self.recording:
                    self.frames_buffer.append(processed_frame.copy())
                    if len(self.frames_buffer) >= target_fps * 5:  # 录制5秒
                        self.complete_recording()
                
                self.frame_queue.append(processed_frame)
                    
        cap.release()
        
    def process_frame(self, frame, frame_id):
        """处理单帧图像 - 直接使用FallDetectionPipeline的方法"""
        try:
            # 使用管道处理帧
            processed_frame, result = self.pipeline.process_frame(frame, frame_id)
            
            # 检查是否有摔倒事件
            fall_detected = False
            for track_id, fall_info in result['falls'].items():
                if fall_info['fall_flag']:
                    fall_detected = True
                    break
            
            # 如果检测到摔倒且未在录制，开始录制
            if not self.pre_fall_status and fall_detected and not self.recording:
                self.start_recording()
            
            # 如果摔倒事件结束且还在录制，结束录制
            if not fall_detected and self.recording:
                self.complete_recording()

            self.pre_fall_status = fall_detected
                
            return processed_frame, result
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame, {}
        
    def start_recording(self):
        """开始录制摔倒片段"""
        self.recording = True
        self.frames_buffer = list(self.frame_queue)
        self.current_event_id = str(uuid.uuid4())
        
    def complete_recording(self):
        """完成录制并发出信号"""
        self.recording = False
        event_id = self.current_event_id
        frames = self.frames_buffer.copy()
        self.frames_buffer = []
        self.fall_detected.emit(event_id, frames)
        
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

class FallDetectionUI(QMainWindow):
    """主界面"""
    def __init__(self):
        super().__init__()
        
        # 初始化系统模块
        self.db = FallDetectionDB()
        self.storage = VideoStorage(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket_name="fall-detection-videos"
        )
        self.config = Config()  # 初始化配置
        
        # 设置UI
        self.setup_ui()
        self.setup_connections()

        # 初始化时加载历史事件
        self.refresh_events_table()
        
        # 视频线程
        self.video_thread = None
        
    def setup_ui(self):
        """初始化界面"""
        self.setWindowTitle("摔倒检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 视频显示区域
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(800, 600)
        self.video_display.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_display)
        
        # 控制区域
        control_layout = QHBoxLayout()
        
        # 摄像头选择
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("默认摄像头", 0)
        self.camera_selector.addItem("RTSP流", "rtsp")
        
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@ip:port/path")
        self.rtsp_input.setVisible(False)
        
        # 控制按钮
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)
        self.export_btn = QPushButton("导出数据")
        
        control_layout.addWidget(QLabel("视频源:"))
        control_layout.addWidget(self.camera_selector)
        control_layout.addWidget(self.rtsp_input)
        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(control_layout)
        
        # 事件记录表格
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(5)
        self.events_table.setHorizontalHeaderLabels([
            "事件ID", "时间", "摄像头", "状态", "操作"
        ])
        self.events_table.setColumnWidth(0, 200)
        self.events_table.setColumnWidth(1, 150)
        self.events_table.setColumnWidth(2, 100)
        self.events_table.setColumnWidth(3, 100)
        self.events_table.setColumnWidth(4, 150)
        main_layout.addWidget(self.events_table)
        
        # 状态栏
        self.statusBar().showMessage("系统就绪")
        
    def setup_connections(self):
        """设置信号槽连接"""
        self.camera_selector.currentIndexChanged.connect(self.update_camera_input)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.export_btn.clicked.connect(self.export_data)
        
    def update_camera_input(self, index):
        """更新摄像头输入选项"""
        is_rtsp = self.camera_selector.itemData(index) == "rtsp"
        self.rtsp_input.setVisible(is_rtsp)
        
    def start_detection(self):
        """开始检测"""
        camera_type = self.camera_selector.currentData()
        
        if camera_type == "rtsp":
            rtsp_url = self.rtsp_input.text().strip()
            if not rtsp_url:
                QMessageBox.warning(self, "输入错误", "请输入有效的RTSP URL")
                return
            source = rtsp_url
        else:
            source = camera_type
            
        # 创建并启动视频线程
        self.video_thread = VideoCaptureThread(source, self.config, self)
        self.video_thread.frame_processed.connect(self.update_video_display)
        self.video_thread.fall_detected.connect(self.handle_fall_event)
        self.video_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("检测运行中...")
        
    def stop_detection(self):
        """停止检测"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("检测已停止")
        
    def update_video_display(self, frame):
        """更新视频显示"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(qt_image))
        
    def handle_fall_event(self, event_id, frames):
        """处理摔倒事件"""
        # 生成时间戳
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        
        # 保存视频到临时文件
        temp_file = os.path.join(tempfile.gettempdir(), f"{event_id}.mp4")
        self.save_video_frames(frames, temp_file)
        
        # 上传到存储
        try:
            object_name = f"events/{event_id}.mp4"
            self.storage.upload_video(temp_file, object_name)
            video_url = f"{self.storage.bucket_name}/{object_name}"
        except Exception as e:
            QMessageBox.warning(self, "上传错误", f"视频上传失败: {str(e)}")
            video_url = ""
            
        # 保存到数据库
        camera_id = self.camera_selector.currentText()
        self.db.add_event(event_id, timestamp, camera_id, video_url)
        
        # 更新事件表格
        self.refresh_events_table()
        
        # 显示通知
        self.show_fall_alert(event_id, timestamp)
        
    def save_video_frames(self, frames, output_path):
        """保存帧序列为视频文件
        参数:
            frames: 帧序列 (列表 of numpy arrays)
            output_path: 输出文件路径 (推荐使用.mp4扩展名)
        """
        if not frames:
            return
            
        height, width, _ = frames[0].shape
        
        # 尝试使用H.264编解码器，如果不支持则回退到MP4V
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编解码器
        
        # 确保输出路径有.mp4扩展名
        if not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'
        
        out = cv2.VideoWriter(output_path, fourcc, 8, (width, height))
        
        if not out.isOpened():
            raise ValueError("无法创建视频文件，请检查编解码器和路径")
            
        for frame in frames:
            out.write(frame)
            
        out.release()
        
    def refresh_events_table(self):
        """刷新事件表格"""
        events = self.db.get_events()
        self.events_table.setRowCount(len(events))
        
        for row, event in enumerate(events):
            event_id, timestamp, camera_id, _, status, _, _ = event
            
            # 填充表格项
            self.events_table.setItem(row, 0, QTableWidgetItem(event_id))
            self.events_table.setItem(row, 1, QTableWidgetItem(timestamp))
            self.events_table.setItem(row, 2, QTableWidgetItem(str(camera_id)))
            self.events_table.setItem(row, 3, QTableWidgetItem(status))
            
            # 添加查看/处理按钮
            view_btn = QPushButton("查看详情/处理事件")
            view_btn.clicked.connect(lambda _, eid=event_id: self.show_event_details(eid))
            self.events_table.setCellWidget(row, 4, view_btn)
            
    def show_event_details(self, event_id):
        """显示事件详情"""
        event, keypoints = self.db.get_event_details(event_id)
        if not event:
            QMessageBox.warning(self, "错误", "未找到该事件")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"事件详情 - {event_id}")
        layout = QVBoxLayout()
        
        # 基本信息
        info_text = f"""
        <b>事件ID:</b> {event[0]}<br>
        <b>时间:</b> {event[1]}<br>
        <b>摄像头:</b> {event[2]}<br>
        <b>状态:</b> {event[4]}<br>
        """
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        # 视频播放
        if event[3]:  # 有视频路径
            try:
                # 下载视频到临时文件
                temp_path = os.path.join(tempfile.gettempdir(), f"playback_{event_id}.mp4")
                self.storage.download_video(event[3], temp_path)
                
                # 设置视频播放器
                video_widget = QVideoWidget()
                video_widget.setMinimumSize(640, 480)
                player = QMediaPlayer()
                player.setVideoOutput(video_widget)
                player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_path)))
                
                # 控制按钮
                controls = QHBoxLayout()
                play_btn = QPushButton("播放")
                play_btn.clicked.connect(player.play)
                pause_btn = QPushButton("暂停")
                pause_btn.clicked.connect(player.pause)
                stop_btn = QPushButton("停止")
                stop_btn.clicked.connect(player.stop)
                
                controls.addWidget(play_btn)
                controls.addWidget(pause_btn)
                controls.addWidget(stop_btn)
                
                layout.addWidget(video_widget)
                layout.addLayout(controls)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法加载视频: {str(e)}")
        
        # 关键点数据
        if keypoints:
            kp_label = QLabel(f"<b>关键点数据帧数:</b> {len(keypoints)}")
            layout.addWidget(kp_label)
        
        # 添加状态切换按钮
        btn_layout = QHBoxLayout()
        
        if event[4] == "未处理":
            mark_processed_btn = QPushButton("标记为已处理")
            mark_processed_btn.clicked.connect(lambda: self.update_event_status(
                event_id, "已处理", dialog))
            btn_layout.addWidget(mark_processed_btn)
        else:
            undo_processed_btn = QPushButton("撤销处理")
            undo_processed_btn.clicked.connect(lambda: self.update_event_status(
                event_id, "未处理", dialog))
            btn_layout.addWidget(undo_processed_btn)
        
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def update_event_status(self, event_id, new_status, dialog=None):
        """更新事件状态通用方法"""
        try:
            self.db.update_event_status(event_id, new_status)
            QMessageBox.information(self, "成功", f"状态已更新为{new_status}")
            
            if dialog:
                dialog.accept()  # 关闭详情对话框
            self.refresh_events_table()  # 刷新事件表格
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"状态更新失败: {str(e)}")
        
    def show_fall_alert(self, event_id, timestamp):
        """显示摔倒警报"""
        alert = QMessageBox(self)
        alert.setWindowTitle("摔倒检测警报")
        alert.setIcon(QMessageBox.Warning)
        alert.setText(f"检测到摔倒事件!\n时间: {timestamp}\n事件ID: {event_id}")
        
        # 添加按钮
        alert.addButton("确认", QMessageBox.AcceptRole)
        view_btn = alert.addButton("查看详情", QMessageBox.ActionRole)
        
        alert.exec_()
        
        if alert.clickedButton() == view_btn:
            self.show_event_details(event_id)
            
    def export_data(self):
        """导出数据为CSV"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出数据", "", "CSV Files (*.csv)")
            
        if file_name:
            try:
                events = self.db.get_events(limit=1000)  # 获取所有事件
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    # 写标题行
                    f.write("event_id,timestamp,camera_id,video_path,status\n")
                    
                    # 写数据行
                    for event in events:
                        line = ','.join([
                            f'"{event[0]}"',  # event_id
                            f'"{event[1]}"',  # timestamp
                            str(event[2]),     # camera_id
                            f'"{event[3]}"' if event[3] else '',  # video_path
                            f'"{event[4]}"'   # status
                        ])
                        f.write(line + '\n')
                        
                QMessageBox.information(self, "导出成功", f"已导出 {len(events)} 条记录到 {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "导出错误", f"导出失败: {str(e)}")
            
    def closeEvent(self, event):
        """关闭窗口事件"""
        self.stop_detection()
        self.db.close()
        event.accept()

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallDetectionUI()
    window.show()
    sys.exit(app.exec_())