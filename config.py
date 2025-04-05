import os

class Config:
    # 模型路径
    YOLO_MODEL_PATH = "/Users/claire/fall_detection/onnx_model/model.onnx"
    MMPOSE_MODEL_PATH = "/Users/claire/fall_detection/onnx_model/td-hm-034c2b.onnx"
    
    # 视频处理参数
    TARGET_FPS = 8
    
    # 检测参数
    CONFIDENCE_THRESHOLD = 0.1
    KEYPOINT_SCORE_THRESHOLD = 0.3
    
    # 摔倒检测参数
    FALL_FRAMES_THRESHOLD = 3
    RECOVERY_FRAMES_THRESHOLD = 3
    RAPID_FALL_DISTANCE_RATIO = 0.15
    RAPID_FALL_FRAME_WINDOW = 10
    
    @staticmethod
    def setup_output_folder(video_path, output_base):
        """设置输出文件夹结构"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_base, video_name)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder