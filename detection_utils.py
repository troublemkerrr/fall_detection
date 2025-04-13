import numpy as np
import math
from typing import List, Tuple, Dict, Any

class DetectionUtils:
    @staticmethod
    def calculate_iou(box1, box2):
        """计算两个边界框的交并比 (IoU)"""
        # 提取边界框坐标
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]

        # 计算交集区域的坐标
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # 计算交集区域的面积
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # 计算两个边界框的面积
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 计算并集区域的面积
        union_area = box1_area + box2_area - inter_area

        # 计算交并比
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    @staticmethod
    def parse_yolo_output(outputs, original_shape, input_shape, confidence_threshold=0.1):
        """解析 YOLO 输出"""
        output = outputs[0][0]  # 获取第一个输出的第一个 batch
        detections = []

        for box in output:
            x_center, y_center, width, height, confidence = box[:5]
            class_probs = box[5:]
            class_id = np.argmax(class_probs)  # 获取类别ID
            if class_id != 0:  # 只保留person
                continue
            class_score = class_probs[class_id]  # 获取类别置信度

            # 过滤低置信度的检测结果
            if confidence < confidence_threshold:
                continue

            # 将边界框坐标转换回原始图像尺寸
            x1 = int(max(0, (x_center - width / 2) * original_shape[1] / input_shape[1]))
            y1 = int(max(0, (y_center - height / 2) * original_shape[0] / input_shape[0]))
            x2 = int(min((x_center + width / 2) * original_shape[1] / input_shape[1], original_shape[1]))
            y2 = int(min((y_center + height / 2) * original_shape[0] / input_shape[0], original_shape[0]))
            
            if x1 >= x2 or y1 >= y2:
                continue

            detections.append((x1, y1, x2, y2, confidence * class_score))
        
        # 非极大值抑制 (NMS)
        filtered_detections = []
        # if len(detections) > 0:
        #     best_box = detections[0]
        #     filtered_detections.append(best_box)

        # iou_threshold = 0.55
        iou_threshold = 0.05
        while len(detections) > 0:
            # 按置信度排序
            detections.sort(key=lambda x: x[4], reverse=True)
            # 取出置信度最高的框
            best_box = detections[0]
            filtered_detections.append(best_box)
            # 移除已经选中的框
            detections = detections[1:]
            # 移除与当前框 IoU 大于阈值的框
            detections = [box for box in detections if DetectionUtils.calculate_iou(best_box, box) < iou_threshold]

        return filtered_detections

    @staticmethod
    def parse_mmpose_output(heatmaps, bbox, padding, resized_size):
        """解析 MMPose 输出"""
        heatmaps = heatmaps / np.max(heatmaps)  # 归一化热力图

        # 处理热力图
        detectNum, keypointNum, heatmapHeight, heatmapWidth = heatmaps.shape
        reshapeHeatmap = np.reshape(heatmaps, (detectNum, keypointNum, -1))
        heatmapMaxIdx = np.reshape(np.argmax(reshapeHeatmap, 2), (detectNum, keypointNum, 1))
        maxVals = np.reshape(np.amax(reshapeHeatmap, 2), (detectNum, keypointNum, 1))

        # 计算预测的关键点坐标
        preds = np.tile(heatmapMaxIdx, (1, 1, 2))
        preds[:, :, 0] = preds[:, :, 0] % heatmapWidth
        preds[:, :, 1] = preds[:, :, 1] // heatmapWidth
        preds = np.where(np.tile(maxVals, (1, 1, 2)) > 0.0, preds, -1)
        preds = preds.astype(np.float32)

        # 对关键点坐标进行微调
        for n in range(detectNum):
            for k in range(keypointNum):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < heatmapWidth - 1 and 1 < py < heatmapHeight - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += (np.sign(diff) * np.array(0.25))

        # 将关键点坐标映射回原始图像空间
        pad_left, pad_top = padding
        resized_w, resized_h = resized_size

        scaleX = (bbox[2] - bbox[0]) / resized_w
        scaleY = (bbox[3] - bbox[1]) / resized_h
        targetCoordinate = np.ones_like(preds[0])
        targetCoordinate[:, 0] = (preds[0][:, 0] * 192 / heatmapWidth - pad_left) * scaleX + bbox[0]
        targetCoordinate[:, 1] = (preds[0][:, 1] * 256 / heatmapHeight - pad_top) * scaleY + bbox[1]

        # 合并坐标和置信度
        refineOutput = np.expand_dims(np.concatenate((targetCoordinate, maxVals[0]), axis=1), 0)
        return refineOutput

    @staticmethod
    def check_fall_status(kpts: List[Dict[str, float]], bbox: Tuple[float]) -> bool:
        """检查是否摔倒"""
        left_shoulder_y = kpts[5]["y"]
        left_shoulder_x = kpts[5]["x"]
        right_shoulder_y = kpts[6]["y"]
        left_hip_y = kpts[11]["y"]
        left_hip_x = kpts[11]["x"]
        right_hip_y = kpts[12]["y"]
        len_factor = math.sqrt(((left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_x - left_hip_x) ** 2))
        left_foot_y = kpts[15]["y"]
        right_foot_y = kpts[16]["y"]

        dx = int(bbox[2] - bbox[0])
        dy = int(bbox[3]- bbox[1])
        difference = dy - dx
        
        condition1 = left_shoulder_y > left_foot_y - len_factor and left_hip_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_hip_y - (len_factor / 2)
        condition2 = right_shoulder_y > right_foot_y - len_factor and right_hip_y > right_foot_y - (len_factor / 2) and right_shoulder_y > right_hip_y - (len_factor / 2)
        
        return condition1 or condition2 or difference < 0