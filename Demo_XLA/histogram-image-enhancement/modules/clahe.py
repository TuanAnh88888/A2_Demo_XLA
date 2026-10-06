"""
Module: modules/clahe.py
Mục đích:
1. Thực hiện thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization).
2. Cho phép tùy chỉnh Clip Limit và Tile Grid Size linh hoạt.
3. Hỗ trợ cả ảnh Grayscale và ảnh Màu (RGB) thông qua không gian màu LAB hoặc YCrCb
   nhằm nâng cao độ tương phản cục bộ mà không làm sai lệch sắc độ màu.
"""

import cv2
import numpy as np


def apply_clahe(img_np, clip_limit=2.0, tile_grid_size=(8, 8), color_space="LAB"):
    """
    Áp dụng thuật toán CLAHE lên ảnh.
    
    Args:
        img_np (np.ndarray): Mảng ảnh đầu vào (uint8, grayscale hoặc RGB).
        clip_limit (float): Ngưỡng giới hạn độ tương phản (thường từ 0.5 đến 5.0).
                            Giá trị càng cao thì độ tương phản càng mạnh, nhưng có thể tăng nhiễu.
        tile_grid_size (tuple): Kích thước lưới ô vuông chia nhỏ ảnh, ví dụ (8, 8), (4, 4), (16, 16).
        color_space (str): Không gian màu sử dụng cho ảnh màu ('LAB' hoặc 'YCrCb'). Mặc định 'LAB'.
    
    Returns:
        np.ndarray: Ảnh sau khi xử lý CLAHE.
    """
    if img_np is None:
        return None

    # Tạo đối tượng CLAHE của OpenCV
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size)
    )

    # Trường hợp 1: Ảnh Grayscale (1 kênh)
    if img_np.ndim == 2:
        return clahe.apply(img_np)
    
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return clahe.apply(img_np[:, :, 0])

    # Trường hợp 2: Ảnh màu RGB
    # Chỉ áp dụng CLAHE lên kênh độ sáng để bảo toàn thông tin màu sắc
    if color_space.upper() == "LAB":
        # Không gian màu LAB: Kênh L (Lightness - 0..255)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Không gian màu YCrCb: Kênh Y (Luma - 0..255)
        ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return result
