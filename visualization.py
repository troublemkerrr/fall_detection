import cv2
import numpy as np
from typing import List, Dict, Tuple

class Visualization:
    # 定义颜色
    COLOR_GREEN = (0, 255, 0)
    COLOR_ORANGE = (51, 153, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 128, 0)
    COLOR_LIGHT_BLUE = (255, 200, 100)
    
    @staticmethod
    def draw_detection(image, bbox, score, is_fall, falling_flag):
        """在图像上绘制检测结果"""
        x1, y1, x2, y2 = bbox

        # 选择颜色
        if is_fall and falling_flag:
            color = Visualization.COLOR_RED
        elif (is_fall or falling_flag):
            color = Visualization.COLOR_ORANGE
        else:
            color = Visualization.COLOR_GREEN

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        
        # 绘制标签
        label = f"{score:.2f}"
        if is_fall:
            label = f"Fall {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)

    @staticmethod
    def draw_keypoints(image, refineOutput, score_threshold=0.3):
        """在原图上绘制关键点和骨架"""
        keypoints = refineOutput[0][:, :2]  # 取前两列（x, y坐标）
        scores = refineOutput[0][:, 2]      # 取第三列（置信度）

        # 定义骨架连接关系
        skeleton_links = [
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [11, 12],
            [5, 11], [5, 6], [6, 12], [15, 13], [13, 11], [5, 7], [7, 9],
            [16, 14], [14, 12], [6, 8], [8, 10]
        ]

        # 绘制骨架连接线
        for link_id, link in enumerate(skeleton_links):
            start_idx, end_idx = link
            if (scores[start_idx] < score_threshold or scores[end_idx] < score_threshold or
                keypoints[start_idx][0] == -1 or keypoints[start_idx][1] == -1 or
                keypoints[end_idx][0] == -1 or keypoints[end_idx][1] == -1):
                continue

            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            if link_id <= 10:
                color = Visualization.COLOR_BLUE
            elif link_id > 14:
                color = Visualization.COLOR_LIGHT_BLUE
            else:
                color = Visualization.COLOR_GREEN

            cv2.line(image, start_point, end_point, color, thickness=1)

        # 绘制关键点
        for kid, ((x, y), score) in enumerate(zip(keypoints, scores)):
            if score < score_threshold or x == -1 or y == -1:
                continue

            x = int(x)
            y = int(y)

            if kid <= 4:
                color = Visualization.COLOR_BLUE
            elif kid % 2:
                color = Visualization.COLOR_GREEN
            else:
                color = Visualization.COLOR_LIGHT_BLUE

            cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
            cv2.putText(image, f"{score:.2f}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)