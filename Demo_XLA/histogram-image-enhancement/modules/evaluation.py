"""
Module: modules/evaluation.py
Mục đích:
1. Tính toán các chỉ số định lượng đánh giá chất lượng ảnh:
   - Mean (Độ sáng trung bình)
   - Standard Deviation - Std (Độ lệch chuẩn / Tương phản)
   - Entropy (Lượng thông tin / Chi tiết ảnh)
   - PSNR (Tỷ số tín hiệu cực đại trên nhiễu - Peak Signal-to-Noise Ratio)
   - SSIM (Chỉ số tương đồng cấu trúc - Structural Similarity Index)
2. Cung cấp hàm so sánh tổng hợp và đưa ra diễn giải ý nghĩa cho từng chỉ số.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def calculate_mean(img_np):
    """Tính giá trị trung bình mức sáng pixel (Mean)"""
    if img_np is None:
        return 0.0
    return float(np.mean(img_np))


def calculate_std(img_np):
    """Tính độ lệch chuẩn mức sáng (Standard Deviation - phản ánh độ tương phản)"""
    if img_np is None:
        return 0.0
    return float(np.std(img_np))


def calculate_entropy(img_np):
    """
    Tính Entropy Shannon của ảnh:
       H = - sum( p(i) * log2(p(i)) )
    Entropy phản ánh sự phong phú về mặt thông tin và mức độ đa dạng mức xám.
    Entropy càng cao thể hiện ảnh chứa nhiều chi tiết và mức xám phong phú.
    """
    if img_np is None:
        return 0.0

    # Chuyển về ảnh xám để tính entropy mức sáng
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        gray = img_np[:, :, 0]
    else:
        gray = img_np

    # Tính phân bố xác suất p(i)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    total_pixels = gray.size
    prob = hist / total_pixels

    # Chỉ tính cho các phần tử có xác suất > 0
    prob_non_zero = prob[prob > 0]
    entropy_val = -np.sum(prob_non_zero * np.log2(prob_non_zero))

    return float(entropy_val)


def calculate_psnr(img_original, img_enhanced):
    """
    Tính PSNR (Peak Signal-to-Noise Ratio) giữa ảnh gốc và ảnh đã xử lý (đơn vị: dB).
    Nếu 2 ảnh giống hệt nhau, PSNR đạt vô cực (Infinity).
    """
    if img_original is None or img_enhanced is None:
        return 0.0
    
    # Đảm bảo cùng kích thước và số kênh
    if img_original.shape != img_enhanced.shape:
        return 0.0

    mse = np.mean((img_original.astype(np.float64) - img_enhanced.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr_val)


def calculate_ssim(img_original, img_enhanced):
    """
    Tính chỉ số tương đồng cấu trúc SSIM (Structural Similarity Index Measure) [-1, 1].
    Giá trị càng gần 1 thể hiện cấu trúc ảnh càng được giữ nguyên.
    """
    if img_original is None or img_enhanced is None:
        return 0.0

    if img_original.shape != img_enhanced.shape:
        return 0.0

    # Nếu là ảnh màu RGB
    if img_original.ndim == 3 and img_original.shape[2] == 3:
        # Chuyển sang ảnh xám để đo tương đồng cấu trúc độ sáng chuẩn
        gray_orig = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        gray_enh = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
        ssim_val = skimage_ssim(gray_orig, gray_enh, data_range=255)
    else:
        gray_orig = img_original.squeeze()
        gray_enh = img_enhanced.squeeze()
        ssim_val = skimage_ssim(gray_orig, gray_enh, data_range=255)

    return float(ssim_val)


def evaluate_all(img_original, img_enhanced):
    """
    Tổng hợp toàn bộ các chỉ số đánh giá trước và sau khi xử lý:
    - Mean
    - Standard Deviation (Std)
    - Entropy
    - PSNR (dB)
    - SSIM (0 -> 1)
    
    Returns:
        dict: Chứa các giá trị trước/sau và chênh lệch phần trăm, cùng lời diễn giải chi tiết.
    """
    if img_original is None or img_enhanced is None:
        return {}

    orig_mean = calculate_mean(img_original)
    enh_mean = calculate_mean(img_enhanced)

    orig_std = calculate_std(img_original)
    enh_std = calculate_std(img_enhanced)

    orig_entropy = calculate_entropy(img_original)
    enh_entropy = calculate_entropy(img_enhanced)

    psnr_val = calculate_psnr(img_original, img_enhanced)
    ssim_val = calculate_ssim(img_original, img_enhanced)

    # Đưa ra nhận xét định tính
    notes = []
    
    # Nhận xét về độ tương phản
    if enh_std > orig_std:
        diff_std = enh_std - orig_std
        notes.append(f"Độ lệch chuẩn tăng (+{diff_std:.2f}): Độ tương phản tổng thể được cải thiện rõ rệt, ranh giới sáng tối sắc nét hơn.")
    elif enh_std < orig_std:
        notes.append(f"Độ lệch chuẩn giảm: Tương phản tổng thể giảm, ảnh có xu hướng phẳng và dịu mắt hơn.")
    else:
        notes.append("Độ tương phản giữ nguyên.")

    # Nhận xét về Entropy
    if enh_entropy > orig_entropy:
        notes.append(f"Entropy tăng từ {orig_entropy:.2f} lên {enh_entropy:.2f}: Lượng thông tin và chi tiết ẩn trong các vùng mức xám được bộc lộ phong phú hơn.")
    elif enh_entropy < orig_entropy:
        notes.append(f"Entropy giảm: Một số mức xám bị gộp cụm hoặc bão hòa sáng/tối.")
    else:
        notes.append("Entropy không thay đổi đáng kể.")

    # Nhận xét về SSIM
    if ssim_val > 0.85:
        notes.append(f"SSIM đạt mức cao ({ssim_val:.3f}): Cấu trúc chi tiết của đối tượng gốc được bảo toàn rất tốt.")
    elif ssim_val > 0.60:
        notes.append(f"SSIM ở mức trung bình khá ({ssim_val:.3f}): Đã có sự thay đổi đáng kể về cường độ sáng để phục vụ tăng cường chất lượng thị giác.")
    else:
        notes.append(f"SSIM thấp ({ssim_val:.3f}): Ảnh đã trải qua biến đổi lớn về phân bố cường độ sáng so với ảnh gốc.")

    return {
        "original": {
            "mean": round(orig_mean, 2),
            "std": round(orig_std, 2),
            "entropy": round(orig_entropy, 3)
        },
        "enhanced": {
            "mean": round(enh_mean, 2),
            "std": round(enh_std, 2),
            "entropy": round(enh_entropy, 3)
        },
        "comparison": {
            "mean_diff": round(enh_mean - orig_mean, 2),
            "std_diff": round(enh_std - orig_std, 2),
            "entropy_diff": round(enh_entropy - orig_entropy, 3),
            "psnr": "Vô cực (∞)" if np.isinf(psnr_val) else f"{psnr_val:.2f} dB",
            "psnr_raw": 999.0 if np.isinf(psnr_val) else round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        },
        "notes": notes
    }
