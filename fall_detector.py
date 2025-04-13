import math
from typing import Dict, Tuple

class FallDetector:
    def __init__(self, config):
        self.config = config
        # 为每个track_id维护独立的状态
        self.track_states: Dict[int, dict] = {}  # {track_id: state_dict}
    
    def _init_track_state(self, track_id: int):
        """初始化一个新track的状态"""
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                'prev_center': None,
                'prev_bbox_diagonal': None,
                'fall_flag': False,
                'fall_frames': 0,
                'recovery_frames': 0,
                'rapid_fall': False,
                'rapid_fall_frame_id': None
            }
    
    def detect_fall(self, track_id: int, current_bbox: Tuple[float], 
                   is_falling_current_frame: bool, current_frame_id: int) -> Tuple[bool, bool]:
        """
        摔倒检测函数（支持多目标）
        
        参数:
        track_id: 目标ID
        current_bbox: 当前帧的人体边界框 (x1, y1, x2, y2)
        is_falling_current_frame: 当前帧是否被判定为摔倒姿态
        current_frame_id: 当前帧号
        
        返回:
        Tuple[bool, bool]: (是否确认摔倒事件, 是否当前帧有摔倒迹象)
        """
        self._init_track_state(track_id)
        state = self.track_states[track_id]
        
        # 计算当前帧的中心点和边界框对角线长度
        current_center = ((current_bbox[0] + current_bbox[2]) / 2, 
                         (current_bbox[1] + current_bbox[3]) / 2)
        current_diagonal = math.sqrt((current_bbox[2] - current_bbox[0])**2 + 
                          (current_bbox[3] - current_bbox[1])**2)
        
        # 快速下坠检测
        if state['prev_center'] is not None and state['prev_bbox_diagonal'] is not None:
            move_distance = math.sqrt((current_center[0] - state['prev_center'][0])**2 + 
                                     (current_center[1] - state['prev_center'][1])**2)
            
            if (move_distance > self.config.RAPID_FALL_DISTANCE_RATIO * state['prev_bbox_diagonal'] 
                and current_center[1] > state['prev_center'][1]):
                state['rapid_fall'] = True
                state['rapid_fall_frame_id'] = current_frame_id
        
        # 更新状态
        state['prev_center'] = current_center
        state['prev_bbox_diagonal'] = current_diagonal
        
        # 摔倒状态判断
        if state['fall_flag']:
            if not is_falling_current_frame:
                state['recovery_frames'] += 1
                if state['recovery_frames'] >= self.config.RECOVERY_FRAMES_THRESHOLD:
                    state['fall_flag'] = False
                    state['recovery_frames'] = 0
                    state['rapid_fall'] = False
            else:
                state['recovery_frames'] = 0
        else:
            if is_falling_current_frame:
                state['fall_frames'] += 1
                if (state['fall_frames'] >= self.config.FALL_FRAMES_THRESHOLD and 
                    state['rapid_fall'] and 
                    state['rapid_fall_frame_id'] is not None and 
                    current_frame_id - state['rapid_fall_frame_id'] < self.config.RAPID_FALL_FRAME_WINDOW):
                    state['fall_flag'] = True
                    state['fall_frames'] = 0
            else:
                state['fall_frames'] = 0
        
        return state['fall_flag'], (is_falling_current_frame or state['rapid_fall_frame_id'] == current_frame_id)