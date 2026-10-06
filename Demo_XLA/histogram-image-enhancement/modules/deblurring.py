"""
Module: modules/deblurring.py
Mục đích:
1. Phục hồi và cải thiện ảnh mờ (Deblurring) ở mức cơ bản.
2. Hỗ trợ các phương pháp:
   - Unsharp Masking cải tiến
   - Richardson-Lucy Deconvolution (Phục hồi mờ Gaussian)
   - Wiener Deconvolution (Phục hồi xấp xỉ mờ chuyển động hoặc nhiễu Gaussian)
3. Cảnh báo khoa học về giới hạn phục hồi khi ảnh bị mất thông tin nghiêm trọng.
"""

import cv2
import numpy as np
from skimage import restoration


def deblur_image(img_np, blur_type="Mờ nhẹ (Unsharp)", iterations=10):
    """
    Cải thiện ảnh bị mờ.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
        blur_type (str): "Mờ nhẹ (Unsharp)", "Mờ Gaussian (Richardson-Lucy)", "Mờ chuyển động (Wiener)"
        iterations (int): Số vòng lặp lặp lại cho Richardson-Lucy (mặc định 10-15)
    Returns:
        np.ndarray: Ảnh sau phục hồi (uint8)
        str: Ghi chú đánh giá khoa học
    """
    if img_np is None:
        return None, ""

    is_color = (img_np.ndim == 3 and img_np.shape[2] == 3)

    if blur_type == "Mờ nhẹ (Unsharp)":
        # Unsharp Mask tăng cường cạnh tần số cao
        blurred = cv2.GaussianBlur(img_np, (0, 0), 2.0)
        deblurred = cv2.addWeighted(img_np, 1.8, blurred, -0.8, 0)
        note = "Đã áp dụng Unsharp Masking để tăng tương phản cục bộ các cạnh mờ."
        return np.clip(deblurred, 0, 255).astype(np.uint8), note

    elif blur_type == "Mờ Gaussian (Richardson-Lucy)":
        # Tạo hàm truyền điểm PSF (Point Spread Function) Gaussian
        psf_size = 5
        psf = cv2.getGaussianKernel(psf_size, 1.2)
        psf = psf @ psf.T

        # Chuẩn hóa về float [0, 1]
        img_float = img_np.astype(np.float64) / 255.0

        if is_color:
            # Xử lý trên từng kênh màu
            deblurred_channels = []
            for ch in range(3):
                deconv = restoration.richardson_lucy(img_float[:, :, ch], psf, num_iter=iterations, clip=False)
                deblurred_channels.append(deconv)
            deblurred = np.stack(deblurred_channels, axis=-1)
        else:
            gray_float = img_float if img_float.ndim == 2 else img_float[:, :, 0]
            deblurred = restoration.richardson_lucy(gray_float, psf, num_iter=iterations, clip=False)

        note = f"Đã giải tích chập Richardson-Lucy ({iterations} vòng lặp) với hàm PSF Gaussian."
        return np.clip(deblurred * 255.0, 0, 255).astype(np.uint8), note

    elif blur_type == "Mờ chuyển động (Wiener)":
        # Tạo PSF chuyển động tuyến tính (Motion Blur PSF)
        psf_size = 7
        psf = np.zeros((psf_size, psf_size), dtype=np.float64)
        psf[int((psf_size - 1) / 2), :] = np.ones(psf_size)
        psf /= psf_size

        img_float = img_np.astype(np.float64) / 255.0

        if is_color:
            deblurred_channels = []
            for ch in range(3):
                deconv = restoration.wiener(img_float[:, :, ch], psf, balance=0.08, clip=False)
                deblurred_channels.append(deconv)
            deblurred = np.stack(deblurred_channels, axis=-1)
        else:
            gray_float = img_float if img_float.ndim == 2 else img_float[:, :, 0]
            deblurred = restoration.wiener(gray_float, psf, balance=0.08, clip=False)

        note = "Đã áp dụng bộ lọc Wiener giải tích chập chuyển động tuyến tính."
        return np.clip(deblurred * 255.0, 0, 255).astype(np.uint8), note

    return img_np.copy(), ""
