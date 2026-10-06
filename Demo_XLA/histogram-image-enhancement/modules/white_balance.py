"""
Module: modules/white_balance.py
Mục đích:
1. Cân bằng trắng (White Balance) khắc phục hiện tượng ám màu (quá vàng, quá xanh, ám đỏ).
2. Hỗ trợ 2 thuật toán:
   - Giả định thế giới xám (Gray World Assumption)
   - Điểm trắng hoàn hảo (White Patch / Max RGB)
"""

import cv2
import numpy as np


def apply_white_balance(img_np, method="Gray World"):
    """
    Áp dụng thuật toán cân bằng trắng tự động.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        method (str): "Gray World" hoặc "White Patch"
    Returns:
        np.ndarray: Ảnh sau cân bằng trắng (uint8)
    """
    if img_np is None:
        return None
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img_np.copy()  # Không áp dụng cho ảnh xám đơn sắc

    img_float = img_np.astype(np.float32)

    if method == "Gray World":
        # Giả thuyết Gray World: Mức trung bình của các kênh R, G, B trong ảnh tự nhiên có xu hướng bằng nhau
        mean_r = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_b = np.mean(img_float[:, :, 2])

        mean_gray = (mean_r + mean_g + mean_b) / 3.0

        scale_r = mean_gray / (mean_r + 1e-5)
        scale_g = mean_gray / (mean_g + 1e-5)
        scale_b = mean_gray / (mean_b + 1e-5)

        balanced_r = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
        balanced_g = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
        balanced_b = np.clip(img_float[:, :, 2] * scale_b, 0, 255)

        return np.stack([balanced_r, balanced_g, balanced_b], axis=-1).astype(np.uint8)

    else:  # White Patch (Perfect Reflector)
        # Giả thuyết: Điểm sáng nhất trong ảnh đại diện cho màu trắng chuẩn (255)
        max_r = np.percentile(img_float[:, :, 0], 99.5)
        max_g = np.percentile(img_float[:, :, 1], 99.5)
        max_b = np.percentile(img_float[:, :, 2], 99.5)

        scale_r = 255.0 / (max_r + 1e-5)
        scale_g = 255.0 / (max_g + 1e-5)
        scale_b = 255.0 / (max_b + 1e-5)

        balanced_r = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
        balanced_g = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
        balanced_b = np.clip(img_float[:, :, 2] * scale_b, 0, 255)

        return np.stack([balanced_r, balanced_g, balanced_b], axis=-1).astype(np.uint8)
