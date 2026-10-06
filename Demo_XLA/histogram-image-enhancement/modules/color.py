"""
Module: modules/color.py
Mục đích:
1. Điều chỉnh độ bão hòa màu sắc (Saturation: 0 -> 200%) qua không gian HSV.
2. Điều chỉnh sắc thái màu (Hue: -180 -> +180 độ).
3. Cân chỉnh nhiệt độ màu (Temperature: Ấm <-> Lạnh).
4. Chuyển đổi mức xám (Grayscale).
5. Hiệu ứng màu cổ điển (Sepia Tone).
"""

import cv2
import numpy as np


def adjust_saturation(img_np, saturation_percent=100):
    """
    Điều chỉnh độ bão hòa màu sắc (Saturation) theo phần trăm.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        saturation_percent (int): 0% (trắng đen hoàn toàn) -> 100% (gốc) -> 200% (rực rỡ)
    """
    if img_np is None:
        return None
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img_np.copy()  # Ảnh xám không có kênh màu để bão hòa

    factor = float(saturation_percent) / 100.0
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_hue(img_np, hue_shift=0):
    """
    Dịch chuyển vòng tròn sắc thái màu (Hue).
    hue_shift trong khoảng [-90, +90] (tương ứng OpenCV H in [0, 179]).
    """
    if img_np is None or hue_shift == 0:
        return img_np.copy() if img_np is not None else None
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img_np.copy()

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(hue_shift)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_temperature(img_np, temp_val=0):
    """
    Điều chỉnh nhiệt độ màu (Color Temperature).
    temp_val: [-100, +100]
    - temp_val > 0: Tông ấm (tăng Đỏ/Vàng, giảm Lam)
    - temp_val < 0: Tông lạnh (tăng Lam, giảm Đỏ)
    """
    if img_np is None or temp_val == 0:
        return img_np.copy() if img_np is not None else None
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return img_np.copy()

    img_float = img_np.astype(np.float32)
    shift = float(temp_val) * 0.4

    if temp_val > 0:
        # Tông ấm: R tăng, B giảm
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] + shift, 0, 255)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] - shift * 0.6, 0, 255)
    else:
        # Tông lạnh: B tăng, R giảm
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] - shift, 0, 255)
        img_float[:, :, 0] = np.clip(img_float[:, :, 0] + shift * 0.6, 0, 255)

    return img_float.astype(np.uint8)


def apply_sepia(img_np):
    """
    Áp dụng bộ lọc màu cổ điển Sepia Tone.
    """
    if img_np is None:
        return None
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        # Chuyển ảnh xám thành 3 kênh trước khi áp dụng sepia
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    transformed = cv2.transform(img_np, sepia_matrix)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def convert_to_grayscale(img_np):
    """
    Chuyển đổi ảnh sang mức xám và định dạng lại thành 3 kênh đồng nhất để hiển thị.
    """
    if img_np is None:
        return None
    if img_np.ndim == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
