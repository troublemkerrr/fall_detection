import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

class SORTTracker:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        """
        基于SORT算法的追踪器
        
        参数:
            max_age: 目标丢失后保留的最大帧数
            min_hits: 新目标被确认前需要连续匹配的最小帧数
            iou_threshold: 匹配的IoU阈值
        """
        self.trackers = []  # 存储所有活跃的追踪器
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1  # 下一个分配的ID

    def update(self, detections: List[List[float]]) -> Dict[int, Dict]:
        """
        更新追踪器状态
        
        参数:
            detections: 当前帧检测到的边界框列表 [[x1,y1,x2,y2,score], ...]
            
        返回:
            更新后的追踪对象 {track_id: {'bbox': [x1,y1,x2,y2], 'score': float, 'tracker': KalmanFilter}}
        """
        self.frame_count += 1
        
        # 存储当前帧的结果
        ret = {}
        
        # 步骤1: 预测所有现有追踪器的位置
        for trk in self.trackers:
            trk['bbox'] = self._predict(trk['tracker'])
            trk['age'] += 1
            trk['is_predicted'] = True
            trk['time_since_update'] += 1
        
        # 步骤2: 数据关联 (匈牙利算法)
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections)
        
        # 步骤3: 更新匹配的追踪器
        for t, d in matched:
            self.trackers[t]['tracker'] = self._update_kalman(self.trackers[t]['tracker'], detections[d][:4])
            self.trackers[t]['bbox'] = detections[d][:4]
            self.trackers[t]['score'] = detections[d][4]
            self.trackers[t]['age'] = 0
            self.trackers[t]['hits'] += 1
            self.trackers[t]['is_predicted'] = False  # 标记为检测框
            self.trackers[t]['time_since_update'] = 0
        
        # 步骤4: 创建新追踪器 (未匹配的检测)
        for i in unmatched_dets:
            trk = {
                'id': self.next_id,
                'bbox': detections[i][:4],
                'score': detections[i][4],
                'tracker': self._init_kalman(detections[i][:4]),
                'age': 0,
                'hits': 1,
                'time_since_update': 0
            }
            self.trackers.append(trk)
            self.next_id += 1
        
        # 步骤5: 删除丢失的追踪器
        final_trackers = []
        for trk in self.trackers:
            # 仅返回已确认的追踪 (hits >= min_hits) 或新追踪但未超过max_age
            if (trk['time_since_update'] < 1) and (trk['hits'] >= self.min_hits or self.frame_count <= self.min_hits):
                ret[trk['id']] = {
                    'bbox': trk['bbox'],
                    'score': trk['score'],
                    'tracker': trk['tracker'],
                    'is_predicted': trk.get('is_predicted', True)  # 未匹配的追踪器默认为预测框
                }
            
            # 移除丢失时间过长的追踪器
            if trk['time_since_update'] < self.max_age:
                final_trackers.append(trk)
        
        self.trackers = final_trackers
        return ret
    
    def _init_kalman(self, bbox: List[float]) -> Dict:
        """
        初始化卡尔曼滤波器
        """
        # 简化的卡尔曼滤波器实现 (实际应用中可以使用filterpy等库)
        # 这里返回一个字典模拟滤波器状态
        return {
            'mean': np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0]),  # [x1,y1,x2,y2,dx,dy]
            'covariance': np.eye(6)  # 初始协方差矩阵
        }
    
    def _predict(self, tracker: Dict) -> List[float]:
        """
        卡尔曼滤波预测步骤
        """
        # 简化的预测: 假设匀速运动
        dt = 1.0  # 时间步长
        F = np.array([
            [1, 0, 0, 0, dt, 0],
            [0, 1, 0, 0, 0, dt],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 更新状态
        tracker['mean'] = F.dot(tracker['mean'])
        tracker['covariance'] = F.dot(tracker['covariance']).dot(F.T)
        
        return tracker['mean'][:4].tolist()
    
    def _update_kalman(self, tracker: Dict, bbox: List[float]) -> Dict:
        """
        卡尔曼滤波更新步骤
        """
        # 简化的更新: 直接使用观测值
        tracker['mean'][:4] = np.array(bbox)
        return tracker
    
    def _associate_detections_to_trackers(self, detections: List[List[float]]) -> Tuple:
        """
        使用匈牙利算法关联检测和追踪器
        """
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._iou(trk['bbox'], det[:4])
        
        # 使用匈牙利算法匹配
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # 处理未匹配的追踪器和检测
        unmatched_trackers = []
        unmatched_detections = []
        
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 0]:
                unmatched_trackers.append(t)
        
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 1]:
                unmatched_detections.append(d)
        
        # 过滤低IoU的匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的IoU
        """
        # 计算交集区域
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])
        
        # 计算交集面积
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0