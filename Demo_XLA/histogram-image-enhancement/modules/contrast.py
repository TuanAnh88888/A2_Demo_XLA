"""
Module: modules/contrast.py
Mục đích:
1. Điều chỉnh độ tương phản của ảnh theo công thức:
      I_new = alpha * I
   - alpha < 1.0: Giảm tương phản (thu hẹp khoảng cách giữa các mức xám)
   - alpha = 1.0: Giữ nguyên
   - alpha > 1.0: Tăng tương phản (mở rộng khoảng cách giữa các mức xám)
2. Kết hợp điều chỉnh đồng thời Độ sáng (Brightness) và Độ tương phản (Contrast).
   Thứ tự chuẩn hóa: Ảnh gốc -> Điều chỉnh Brightness -> Điều chỉnh Contrast -> Ảnh kết quả.
"""

import cv2
import numpy as np


def adjust_contrast(img_np, alpha=1.0):
    """
    Điều chỉnh độ tương phản của ảnh.
    
    Args:
        img_np (np.ndarray): Mảng ảnh đầu vào (uint8, grayscale hoặc RGB).
        alpha (float): Hệ số nhân tương phản, dải giá trị [0.1, 3.0].
                       - alpha < 1.0: Giảm tương phản
                       - alpha = 1.0: Giữ nguyên
                       - alpha > 1.0: Tăng tương phản
    
    Returns:
        np.ndarray: Ảnh sau khi điều chỉnh tương phản (uint8).
    """
    if img_np is None:
        return None

    # Nhân hệ số alpha ở kiểu float32 rồi clip chặn về [0, 255]
    adjusted = np.clip(float(alpha) * img_np.astype(np.float32), 0, 255).astype(np.uint8)

    return adjusted


def adjust_brightness_contrast(img_np, alpha=1.0, beta=0):
    """
    Kết hợp điều chỉnh đồng thời Độ sáng (Brightness) và Độ tương phản (Contrast).
    
    Theo đúng quy trình:
      Ảnh gốc -> Điều chỉnh Brightness (+ beta) -> Điều chỉnh Contrast (* alpha) -> Ảnh kết quả
      
    Args:
        img_np (np.ndarray): Ảnh đầu vào (uint8).
        alpha (float): Hệ số tương phản [0.1, 3.0].
        beta (int hoặc float): Độ lệch mức sáng [-100, 100].
        
    Returns:
        np.ndarray: Ảnh sau khi xử lý kết hợp cả hai yếu tố.
    """
    if img_np is None:
        return None

    # Bước 1: Điều chỉnh Brightness trước
    bright_step = np.clip(img_np.astype(np.int32) + int(beta), 0, 255).astype(np.float32)

    # Bước 2: Điều chỉnh Contrast trên kết quả bước 1
    contrast_step = np.clip(float(alpha) * bright_step, 0, 255).astype(np.uint8)

    return contrast_step
