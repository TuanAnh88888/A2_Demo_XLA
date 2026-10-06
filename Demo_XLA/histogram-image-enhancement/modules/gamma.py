"""
Module: modules/gamma.py
Mục đích:
1. Hiệu chỉnh Gamma (Gamma Correction) phi tuyến: I' = 255 * (I / 255)^gamma
2. Điều chỉnh mức độ phơi sáng (Exposure Value - EV)
3. Điều chỉnh vùng sáng mạnh (Highlights) và vùng tối (Shadows) độc lập
"""

import cv2
import numpy as np


def adjust_gamma(img_np, gamma=1.0):
    """
    Hiệu chỉnh Gamma phi tuyến tính.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        gamma (float): Hệ số gamma (0.1 -> 3.0).
                       gamma < 1.0: Làm sáng vùng tối mà ít làm cháy vùng sáng
                       gamma = 1.0: Giữ nguyên
                       gamma > 1.0: Làm tối ảnh và nén mức sáng
    Returns:
        np.ndarray: Ảnh sau khi hiệu chỉnh gamma (uint8)
    """
    if img_np is None:
        return None

    gamma = max(0.01, float(gamma))
    # Sử dụng bảng tra cứu LUT (Look-Up Table) để tối ưu hóa hiệu năng
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img_np, table)


def adjust_exposure(img_np, ev=0.0):
    """
    Điều chỉnh mức phơi sáng (Exposure) theo thang EV (-2.0 đến +2.0).
    Công thức: I' = I * (2 ^ ev)
    """
    if img_np is None:
        return None

    ev = float(ev)
    if ev == 0.0:
        return img_np.copy()

    scale = 2.0 ** ev
    adjusted = np.clip(img_np.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return adjusted


def adjust_highlights_shadows(img_np, highlights=0, shadows=0):
    """
    Điều chỉnh độc lập vùng sáng (Highlights) và vùng tối (Shadows).
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        highlights (int): Điều chỉnh vùng sáng (-100 đến +100)
        shadows (int): Điều chỉnh vùng tối (-100 đến +100)
    """
    if img_np is None:
        return None
    if highlights == 0 and shadows == 0:
        return img_np.copy()

    img_float = img_np.astype(np.float32) / 255.0
    
    # Tính kênh độ sáng đại diện
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        luma = (0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2])[:, :, None]
    else:
        luma = img_float

    # Trọng số làm mượt cho vùng tối (Shadows tập trung vào luma thấp)
    shadow_weight = np.clip(1.0 - (luma / 0.5), 0.0, 1.0) ** 2
    # Trọng số làm mượt cho vùng sáng (Highlights tập trung vào luma cao)
    highlight_weight = np.clip((luma - 0.5) / 0.5, 0.0, 1.0) ** 2

    delta_shadow = (shadows / 100.0) * 0.4 * shadow_weight
    delta_highlight = (highlights / 100.0) * 0.4 * highlight_weight

    result = img_float + delta_shadow + delta_highlight
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
