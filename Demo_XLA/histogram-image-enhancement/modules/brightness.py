"""
Module: modules/brightness.py
Mục đích:
1. Điều chỉnh độ sáng của ảnh theo công thức tuyến tính:
      I_new = I + beta
2. Giới hạn giá trị pixel trong khoảng chuẩn [0, 255] để tránh tràn số (clipping/overflow).
3. Hỗ trợ dải beta từ -100 (ảnh tối đi) đến +100 (ảnh sáng lên).
"""

import cv2
import numpy as np


def adjust_brightness(img_np, beta=0):
    """
    Điều chỉnh độ sáng của ảnh.
    
    Args:
        img_np (np.ndarray): Mảng ảnh đầu vào (uint8, grayscale hoặc RGB).
        beta (int hoặc float): Giá trị thay đổi độ sáng, dải giá trị [-100, 100].
                               - beta < 0: Ảnh tối dần
                               - beta = 0: Giữ nguyên
                               - beta > 0: Ảnh sáng dần
    
    Returns:
        np.ndarray: Ảnh sau khi điều chỉnh độ sáng (uint8, cùng kích thước và số kênh).
    """
    if img_np is None:
        return None

    # Ép kiểu sang int32 hoặc float32 để cộng beta không bị tràn số uint8 (dưới 0 hoặc trên 255)
    # Sau đó dùng np.clip để chặn trong khoảng [0, 255] rồi ép lại uint8
    adjusted = np.clip(img_np.astype(np.int32) + int(beta), 0, 255).astype(np.uint8)

    return adjusted
