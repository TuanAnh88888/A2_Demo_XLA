"""
Module: modules/denoising.py
Mục đích:
1. Khử nhiễu ảnh số (Image Denoising) với các phương pháp:
   - Gaussian Blur (Làm mờ Gauss)
   - Median Filter (Bộ lọc trung vị - khử nhiễu muối tiêu)
   - Bilateral Filter (Lọc song phương - khử nhiễu bảo toàn cạnh sắc nét)
   - Non-Local Means (NL-Means - Khử nhiễu phi cục bộ chất lượng cao)
2. Hỗ trợ 3 cấp độ: Nhẹ, Vừa, Mạnh cho cả ảnh Grayscale và ảnh màu.
"""

import cv2
import numpy as np


def apply_denoising(img_np, method="Bilateral Filter", level="Vừa"):
    """
    Áp dụng thuật toán khử nhiễu lên ảnh.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        method (str): "Gaussian Blur", "Median Filter", "Bilateral Filter", "Non-Local Means"
        level (str): "Nhẹ", "Vừa", "Mạnh"
    Returns:
        np.ndarray: Ảnh sau khi khử nhiễu (uint8)
    """
    if img_np is None:
        return None

    is_color = (img_np.ndim == 3 and img_np.shape[2] == 3)

    # Cấu hình tham số theo 3 cấp độ: Nhẹ / Vừa / Mạnh
    level_params = {
        "Nhẹ": {"ksize": 3, "sigma": 0.8, "d": 5, "sigma_c": 30, "sigma_s": 30, "h": 5},
        "Vừa": {"ksize": 5, "sigma": 1.2, "d": 7, "sigma_c": 50, "sigma_s": 50, "h": 10},
        "Mạnh": {"ksize": 7, "sigma": 2.0, "d": 9, "sigma_c": 75, "sigma_s": 75, "h": 15},
    }
    p = level_params.get(level, level_params["Vừa"])

    if method == "Gaussian Blur":
        return cv2.GaussianBlur(img_np, (p["ksize"], p["ksize"]), sigmaX=p["sigma"])

    elif method == "Median Filter":
        return cv2.medianBlur(img_np, p["ksize"])

    elif method == "Bilateral Filter":
        # Bilateral Filter làm mịn nhiễu vùng phẳng nhưng giữ lại đường biên sắc nét
        return cv2.bilateralFilter(
            img_np,
            d=p["d"],
            sigmaColor=p["sigma_c"],
            sigmaSpace=p["sigma_s"]
        )

    elif method == "Non-Local Means":
        # NL-Means là kỹ thuật khử nhiễu tiên tiến dựa trên tính tự tương đồng cấu trúc
        if is_color:
            return cv2.fastNlMeansDenoisingColored(
                img_np,
                None,
                h=p["h"],
                hColor=p["h"],
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            gray = img_np if img_np.ndim == 2 else img_np[:, :, 0]
            return cv2.fastNlMeansDenoising(
                gray,
                None,
                h=p["h"],
                templateWindowSize=7,
                searchWindowSize=21
            )

    return img_np.copy()
