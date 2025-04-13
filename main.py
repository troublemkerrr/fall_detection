import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from config import Config
from models import YOLOModel, MMPoseModel
from detection_utils import DetectionUtils
from visualization import Visualization
from fall_detector import FallDetector
from simple_tracker import SORTTracker

class FallDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.yolo_model = YOLOModel(config.YOLO_MODEL_PATH)
        self.mmpose_model = MMPoseModel(config.MMPOSE_MODEL_PATH)
        self.fall_detector = FallDetector(config)
        # self.tracker = SimpleTracker(
        #     iou_threshold_range=(0.5, 0.7),
        #     max_miss_frames=5
        # )

        self.tracker = SORTTracker()
    
    def process_video(self, video_path: str, output_folder: str) -> None:
        """处理视频文件"""
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # 获取视频属性
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = min(input_fps, self.config.TARGET_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化 VideoWriter
        output_video_path = os.path.join(output_folder, "output.mp4")
        video_writer = self._init_video_writer(output_video_path, width, height, target_fps)

        # 计算帧率转换比例
        frame_ratio = input_fps / target_fps
        frame_idx = 0
        processed_frame_idx = 0

        results = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1

            # 根据帧率比例跳帧或重复帧
            if frame_idx % frame_ratio > 1:
                continue  # 跳帧

            print(f"Processing frame {frame_idx} (output frame {processed_frame_idx})")

            # 处理单帧
            frame_vis, result = self.process_frame(frame, processed_frame_idx)
            
            # 保存结果
            results.append(result)
            cv2.imwrite(os.path.join(output_folder, f"frame_{processed_frame_idx}.jpg"), frame_vis)
            video_writer.write(frame_vis)
            
            processed_frame_idx += 1

        # 释放资源
        cap.release()
        video_writer.release()

        # 保存结果到JSON
        with open(os.path.join(output_folder, "res.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Video processing completed.")

    def _init_video_writer(self, output_path: str, width: int, height: int, fps: float):
        """初始化视频写入器"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Failed to initialize VideoWriter for {output_path}")
        return writer

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理单帧图像（增加追踪功能）"""
        original_shape = frame.shape[:2]
        frame_vis = frame.copy()

        # 运行YOLO检测
        yolo_outputs = self.yolo_model.detect(frame)
        detections = DetectionUtils.parse_yolo_output(
            yolo_outputs, 
            original_shape, 
            self.yolo_model.input_shape,
            self.config.CONFIDENCE_THRESHOLD
        )

        # 初始化结果结构
        result = {
            'frame_id': frame_id,
            'tracks': {},
            'keypoints': {},
            'falls': {}
        }

        # 如果没有检测到人，直接返回
        if not detections:
            return frame_vis, result

        # 更新追踪器
        tracked_objects = self.tracker.update([det[:5] for det in detections])

        # 处理每个追踪到的人
        for track_id, track_info in tracked_objects.items():
            if track_info['is_predicted']:
                continue
            bbox = track_info['bbox']
            
            # 裁剪人体区域
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[max(0,y1):min(y2,frame.shape[0]), max(0,x1):min(x2,frame.shape[1])]
            
            if cropped.size == 0:
                continue

            # 运行MMPose姿态估计
            heatmaps, original_shape_pose, resized_size, padding = self.mmpose_model.predict(cropped)
            refineOutput = DetectionUtils.parse_mmpose_output(heatmaps, bbox, padding, resized_size)
            
            # 绘制关键点
            Visualization.draw_keypoints(frame_vis, refineOutput, self.config.KEYPOINT_SCORE_THRESHOLD)
            
            # 转换为可序列化的关键点格式
            serializable_output = self._convert_keypoints_to_serializable(refineOutput)
            
            # 检查摔倒状态
            is_fall = DetectionUtils.check_fall_status(serializable_output, bbox)
            
            # 使用状态机检测摔倒
            fall_flag, falling_flag = self.fall_detector.detect_fall(bbox, is_fall, frame_id)
            # is_fall, fall_flag, falling_flag = False, False, False
            
            # 绘制检测框和ID
            color = self._get_track_color(track_id)
            Visualization.draw_tracking_info(frame_vis, bbox, track_id, track_info['score'], 
                                          fall_flag, falling_flag, color)
            
            # 保存结果
            result['tracks'][track_id] = {
                'bbox': bbox,
                'score': track_info['score'],
                'history': track_info.get('history', [])
            }
            result['keypoints'][track_id] = serializable_output
            result['falls'][track_id] = {
                'is_fall': is_fall,
                'fall_flag': fall_flag,
                'falling_flag': falling_flag
            }

        return frame_vis, result
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """根据track_id生成固定颜色"""
        # 使用hash确保相同ID总是返回相同颜色
        h = hash(str(track_id)) % 360
        return Visualization.hsv_to_rgb(h, 1.0, 1.0)

    def _convert_keypoints_to_serializable(self, refineOutput: np.ndarray) -> List[Dict[str, float]]:
        """将关键点转换为可序列化的字典格式"""
        return [{
            'x': float(point[0]),
            'y': float(point[1]),
            'score': float(point[2])
        } for point in refineOutput[0]]

def main():
    config = Config()
    video_path = "/Users/claire/fall_detection/test_data/50_ways_to_fall.mp4"
    # video_path = "/Users/claire/fall_detection/test_data/demo.mp4"
    output_base = "/Users/claire/fall_detection/test_res"
    
    # 设置输出文件夹
    output_folder = Config.setup_output_folder(video_path, output_base)
    
    # 创建处理管道并运行
    pipeline = FallDetectionPipeline(config)
    pipeline.process_video(video_path, output_folder)

if __name__ == "__main__":
    main()