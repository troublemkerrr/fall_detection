import math
from typing import Tuple

class FallDetector:
    def __init__(self, config):
        self.prev_center = None
        self.prev_bbox_diagonal = None
        self.fall_flag = False
        self.fall_frames = 0
        self.recovery_frames = 0
        self.rapid_fall = False
        self.rapid_fall_frame_id = None
        self.config = config

    def detect_fall(self, current_bbox: Tuple[float], is_falling_current_frame: bool, current_frame_id: int) -> Tuple[bool, bool]:
        """
        摔倒检测函数
        
        参数:
        current_bbox: 当前帧的人体边界框 (x1, y1, x2, y2)
        is_falling_current_frame: 当前帧是否被判定为摔倒姿态 (True/False)
        current_frame_id: 当前帧号
        
        返回:
        Tuple[bool, bool]: (是否确认摔倒事件, 是否当前帧有摔倒迹象)
        """
        # 计算当前帧的中心点和边界框对角线长度
        current_center = ((current_bbox[0] + current_bbox[2]) / 2, 
                         (current_bbox[1] + current_bbox[3]) / 2)
        current_diagonal = math.sqrt((current_bbox[2] - current_bbox[0])**2 + 
                          (current_bbox[3] - current_bbox[1])**2)
        
        # 快速下坠检测 (只有有上一帧数据时才检测)
        if self.prev_center is not None and self.prev_bbox_diagonal is not None:
            move_distance = math.sqrt((current_center[0] - self.prev_center[0])**2 + 
                                     (current_center[1] - self.prev_center[1])**2)
            
            # 如果移动距离超过前一帧对角线长度的20%，判定为快速下坠
            if move_distance > self.config.RAPID_FALL_DISTANCE_RATIO * self.prev_bbox_diagonal and current_center[1] > self.prev_center[1]:
                self.rapid_fall = True
                self.rapid_fall_frame_id = current_frame_id
        
        # 更新上一帧信息
        self.prev_center = current_center
        self.prev_bbox_diagonal = current_diagonal
        
        # 摔倒状态判断
        if self.fall_flag:
            # 如果已经处于摔倒状态，检查是否需要恢复
            if not is_falling_current_frame:
                self.recovery_frames += 1
                
                # 连续3帧正常则恢复状态
                if self.recovery_frames >= self.config.RECOVERY_FRAMES_THRESHOLD:
                    self.fall_flag = False
                    self.recovery_frames = 0
                    self.rapid_fall = False
            else:
                self.recovery_frames = 0
        else:
            # 未处于摔倒状态，检查是否需要触发摔倒
            if is_falling_current_frame:
                self.fall_frames += 1

                # 连续3帧满足条件则触发摔倒
                if (self.fall_frames >= self.config.FALL_FRAMES_THRESHOLD and 
                    self.rapid_fall and 
                    self.rapid_fall_frame_id is not None and 
                    current_frame_id - self.rapid_fall_frame_id < self.config.RAPID_FALL_FRAME_WINDOW):
                    self.fall_flag = True
                    self.fall_frames = 0

            else:
                self.fall_frames = 0
        
        return self.fall_flag, (is_falling_current_frame or self.rapid_fall_frame_id == current_frame_id)