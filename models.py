import cv2
import numpy as np
import onnxruntime

class YOLOModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (H, W)
        
    def preprocess(self, image):
        """预处理输入图像"""
        resized_image = cv2.resize(image, self.input_shape)
        input_image = resized_image / 255.0  # 归一化到 [0, 1]
        input_image = input_image.transpose(2, 0, 1)  # 转换维度顺序为 (C, H, W)
        input_image = np.expand_dims(input_image, axis=0)  # 添加 batch 维度
        return input_image.astype(np.float32)
    
    def detect(self, image):
        """运行检测"""
        input_image = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_image})
        return outputs

class MMPoseModel:
    def __init__(self, model_path, target_size=(192, 256)):
        self.session = onnxruntime.InferenceSession(model_path)
        self.target_size = target_size
        
    def preprocess(self, cropped_image):
        """预处理裁剪图像"""
        original_h, original_w = cropped_image.shape[:2]
        
        scale = min(self.target_size[0] / original_w, self.target_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        resized_image = cv2.resize(cropped_image, (new_w, new_h))

        pad_w = self.target_size[0] - new_w
        pad_h = self.target_size[1] - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, 
                                        pad_left, pad_right, 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

        input_image = padded_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(input_image, axis=0), (original_h, original_w), (new_w, new_h), (pad_left, pad_top)
    
    def predict(self, cropped_image):
        """运行姿态估计"""
        input_image, original_shape, resized_size, padding = self.preprocess(cropped_image)
        input_name = self.session.get_inputs()[0].name
        heatmaps = self.session.run(None, {input_name: input_image})[0]
        return heatmaps, original_shape, resized_size, padding