"""
Module: modules/equalization.py
Mục đích:
1. Thực hiện Histogram Equalization bằng OpenCV (cv2.equalizeHist).
2. Xử lý chuẩn xác cho ảnh màu RGB thông qua không gian màu YCrCb (tránh méo màu).
3. Triển khai thuật toán cân bằng Histogram thủ công bằng NumPy (dựa trên hàm CDF)
   để phục vụ giảng dạy lý thuyết và báo cáo khoa học.
4. Minh họa so sánh cân bằng trên kênh Y (chuẩn) vs cân bằng trực tiếp từng kênh RGB (bị biến đổi màu sắc).
"""

import cv2
import numpy as np


def equalize_histogram_opencv(img_np):
    """
    Cân bằng Histogram sử dụng thư viện OpenCV.
    - Với ảnh Grayscale: Áp dụng trực tiếp cv2.equalizeHist()
    - Với ảnh RGB: Chuyển sang không gian màu YCrCb, cân bằng duy nhất kênh Y (Luminance - độ sáng),
      giữ nguyên các kênh màu sắc Cr, Cb để bảo toàn tông màu tự nhiên.
    
    Args:
        img_np (np.ndarray): Ảnh đầu vào (uint8, grayscale hoặc RGB).
    Returns:
        np.ndarray: Ảnh sau khi cân bằng Histogram.
    """
    if img_np is None:
        return None

    # Trường hợp 1: Ảnh mức xám (Grayscale)
    if img_np.ndim == 2:
        return cv2.equalizeHist(img_np)
    
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return cv2.equalizeHist(img_np[:, :, 0])

    # Trường hợp 2: Ảnh màu RGB
    # Bước 1: Chuyển từ RGB sang YCrCb (Y: Độ sáng, Cr: Red-difference, Cb: Blue-difference)
    ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)

    # Bước 2: Tách kênh Y và thực hiện cân bằng Histogram trên kênh Y
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    # Bước 3: Chuyển ngược lại không gian màu RGB
    img_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return img_equalized


def equalize_histogram_numpy(img_np):
    """
    Tự cài đặt thuật toán Histogram Equalization bằng NumPy theo công thức giải tích:
    1. Tính lược đồ mức xám (Histogram) h(v)
    2. Tính hàm phân bố tích lũy (Cumulative Distribution Function - CDF):
       CDF(v) = sum_{i=0}^v h(i)
    3. Ánh xạ mức xám mới:
       h_new(v) = round( (CDF(v) - CDF_min) / ((M * N) - CDF_min) * (L - 1) )
       với L = 256 mức xám.
    
    Returns:
        tuple: (ảnh_kết_quả, cdf_gốc, cdf_chuẩn_hóa, lut_bảng_ánh_xạ)
    """
    if img_np is None:
        return None, None, None, None

    def _equalize_channel_numpy(channel):
        """Hàm phụ trợ cân bằng 1 kênh đơn 2D bằng NumPy"""
        # 1. Tính histogram
        hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])

        # 2. Tính CDF (tổng tích lũy)
        cdf = hist.cumsum()

        # 3. Loại trừ các giá trị 0 khi tìm CDF_min
        cdf_masked = np.ma.masked_equal(cdf, 0)

        # 4. Ánh xạ chuẩn hóa mức xám về [0, 255]
        total_pixels = channel.shape[0] * channel.shape[1]
        cdf_min = cdf_masked.min()
        
        # Tránh chia cho 0 nếu ảnh đơn sắc tuyệt đối
        if total_pixels == cdf_min:
            lut = np.zeros(256, dtype=np.uint8)
        else:
            lut = (cdf_masked - cdf_min) * 255 / (total_pixels - cdf_min)
            lut = np.ma.filled(lut, 0).astype(np.uint8)

        # 5. Gán giá trị mới theo bảng ánh xạ (Lookup Table - LUT)
        equalized_channel = lut[channel]
        return equalized_channel, hist, cdf, lut

    if img_np.ndim == 2:
        eq_img, hist, cdf, lut = _equalize_channel_numpy(img_np)
        return eq_img, cdf, lut
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        eq_img, hist, cdf, lut = _equalize_channel_numpy(img_np[:, :, 0])
        return eq_img, cdf, lut
    else:
        # Ảnh màu: Chuyển sang YCrCb, cân bằng kênh Y bằng NumPy, ghép lại RGB
        ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        eq_y, hist_y, cdf_y, lut_y = _equalize_channel_numpy(ycrcb[:, :, 0])
        ycrcb[:, :, 0] = eq_y
        result_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return result_rgb, cdf_y, lut_y


def equalize_histogram_rgb_direct(img_np):
    """
    Hàm minh họa: Cân bằng Histogram trực tiếp trên cả 3 kênh R, G, B độc lập.
    CHÚ Ý: Phương pháp này làm thay đổi tỷ lệ màu sắc tương đối giữa các kênh,
    dẫn đến hiện tượng lệch màu (color shift / chromatic distortion).
    Được viết để so sánh trực quan với phương pháp chuẩn YCrCb.
    """
    if img_np is None or img_np.ndim != 3:
        return img_np

    channels = cv2.split(img_np)
    eq_channels = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(eq_channels)
