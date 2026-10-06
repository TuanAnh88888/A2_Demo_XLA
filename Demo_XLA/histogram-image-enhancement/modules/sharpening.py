"""
Module: modules/sharpening.py
Mục đích:
1. Làm nét ảnh (Sharpening): Cải thiện ảnh mờ nhẹ, tăng cường độ sắc nét cạnh.
2. Hỗ trợ các phương pháp:
   - Sharpen Kernel (Ma trận lọc không gian)
   - Unsharp Masking (Kỹ thuật mặt nạ làm mờ truyền thống)
   - High-pass Filtering
3. Tăng cường chi tiết (Detail Enhancement) & Tăng cường đường biên (Edge Enhancement).
"""

import cv2
import numpy as np


def sharpen_image(img_np, strength=50, method="Unsharp Mask"):
    """
    Làm nét ảnh với cường độ từ 0 đến 100.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        strength (int): Cường độ làm nét (0 -> 100). 0 là giữ nguyên.
        method (str): "Unsharp Mask", "Kernel Sharpen", hoặc "High-Pass"
    Returns:
        np.ndarray: Ảnh sau khi làm nét (uint8)
    """
    if img_np is None or strength <= 0:
        return img_np.copy() if img_np is not None else None

    weight = float(strength) / 50.0  # Chuẩn hóa về dải [0.0, 2.0]

    if method == "Unsharp Mask":
        # Unsharp Mask: Ảnh gốc + weight * (Ảnh gốc - Bản làm mờ Gaussian)
        blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.5)
        sharp = cv2.addWeighted(img_np, 1.0 + weight, blurred, -weight, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    elif method == "Kernel Sharpen":
        # Nhân chập ma trận Laplace tiêu chuẩn
        # Kernel: [[0, -1, 0], [-1, 4 + k, -1], [0, -1, 0]]
        k = weight * 1.5
        kernel = np.array([
            [0, -k/4, 0],
            [-k/4, 1 + k, -k/4],
            [0, -k/4, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(img_np, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    else:  # High-Pass
        # Tìm thành phần tần số cao (High-pass) qua trừ Gaussian rồi cộng bù
        blurred = cv2.GaussianBlur(img_np, (5, 5), 1.0)
        high_pass = cv2.subtract(img_np, blurred)
        sharp = cv2.addWeighted(img_np, 1.0, high_pass, weight * 1.2, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_details(img_np, strength=50):
    """
    Tăng cường chi tiết bề mặt (Detail Enhancement).
    Sử dụng cv2.detailEnhance cho ảnh màu hoặc Unsharp Mask đa tầng cho ảnh xám.
    """
    if img_np is None or strength <= 0:
        return img_np.copy() if img_np is not None else None

    factor = float(strength) / 100.0

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        # OpenCV detailEnhance hỗ trợ ảnh màu 3 kênh
        sigma_s = 10 + 20 * factor
        sigma_r = 0.15 + 0.3 * factor
        enhanced = cv2.detailEnhance(img_np, sigma_s=sigma_s, sigma_r=sigma_r)
        # Pha trộn theo tỷ lệ cường độ
        return cv2.addWeighted(img_np, 1.0 - factor, enhanced, factor, 0)
    else:
        # Ảnh mức xám: dùng 2 cấp Unsharp Mask
        blur1 = cv2.GaussianBlur(img_np, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(img_np, (7, 7), 2.5)
        detail = cv2.subtract(blur1, blur2)
        res = cv2.addWeighted(img_np, 1.0, detail, factor * 2.0, 0)
        return np.clip(res, 0, 255).astype(np.uint8)


def enhance_edges(img_np, strength=50):
    """
    Tăng cường đường biên cạnh (Edge Enhancement) dựa trên toán tử Sobel.
    """
    if img_np is None or strength <= 0:
        return img_np.copy() if img_np is not None else None

    factor = float(strength) / 100.0 * 0.8

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Tính gradient theo phương x và y
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    edge_mag = np.clip(edge_mag, 0, 255).astype(np.uint8)

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        edge_3ch = cv2.cvtColor(edge_mag, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(img_np, 1.0, edge_3ch, factor, 0)
    else:
        result = cv2.addWeighted(img_np, 1.0, edge_mag, factor, 0)

    return np.clip(result, 0, 255).astype(np.uint8)
