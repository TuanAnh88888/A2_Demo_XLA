"""
Module: modules/auto_enhance.py
Mục đích:
1. Tự động chẩn đoán các khiếm khuyết trên ảnh (thiếu sáng, dư sáng, tương phản thấp, mờ, ám màu).
2. Tự động xây dựng chuỗi giải pháp nâng cao chất lượng tối ưu dựa trên số liệu thực tế.
3. Báo cáo minh bạch các bước đã thực thi và thông số tương ứng.
"""

import cv2
import numpy as np

from modules.histogram import calculate_statistics
from modules.brightness import adjust_brightness
from modules.contrast import adjust_contrast
from modules.gamma import adjust_gamma
from modules.clahe import apply_clahe
from modules.sharpening import sharpen_image
from modules.denoising import apply_denoising
from modules.white_balance import apply_white_balance


def auto_enhance_image(img_np):
    """
    Tự động phân tích và áp dụng chuỗi cải thiện phù hợp nhất cho ảnh hiện tại.
    
    Args:
        img_np (np.ndarray): Ảnh uint8
    Returns:
        tuple: (ảnh_sau_xử_lý, danh_sách_bước_đã_làm, phân_tích_lý_do)
    """
    if img_np is None:
        return None, [], ""

    stats = calculate_statistics(img_np)
    mean_val = stats.get("mean", 128.0)
    std_val = stats.get("std", 50.0)
    dyn_range = stats.get("dynamic_range", 200.0)

    # Ước tính độ mờ bằng phương sai toán tử Laplacian (Laplacian Variance)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Kiểm tra ám màu nếu là ảnh màu
    color_cast = False
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        m_r = np.mean(img_np[:, :, 0])
        m_g = np.mean(img_np[:, :, 1])
        m_b = np.mean(img_np[:, :, 2])
        # Nếu chênh lệch giữa các kênh màu > 25 điểm sáng thì nghi ngờ bị ám màu
        if max(abs(m_r - m_g), abs(m_r - m_b), abs(m_g - m_b)) > 28:
            color_cast = True

    steps = []
    reasons = []
    result = img_np.copy()

    # Bước 1: Khắc phục ám màu nếu có
    if color_cast:
        result = apply_white_balance(result, method="Gray World")
        steps.append("✓ Cân bằng trắng (White Balance - Gray World)")
        reasons.append("Phát hiện độ chênh lệch màu lớn giữa các kênh (nghi ngờ ám màu).")

    # Bước 2: Khắc phục thiếu sáng / dư sáng
    if mean_val < 85:
        # Ảnh thiếu sáng
        beta_add = int(np.clip((110 - mean_val) * 0.5, 10, 40))
        result = adjust_brightness(result, beta=beta_add)
        result = adjust_gamma(result, gamma=0.85)  # Kéo sáng vùng tối
        steps.append(f"✓ Tăng độ sáng (Brightness +{beta_add})")
        steps.append("✓ Hiệu chỉnh Gamma (γ = 0.85)")
        reasons.append(f"Độ sáng trung bình thấp ({mean_val:.1f} < 85): Ảnh bị thiếu sáng.")
    elif mean_val > 175:
        # Ảnh dư sáng
        beta_sub = int(np.clip((mean_val - 150) * 0.4, 10, 30))
        result = adjust_brightness(result, beta=-beta_sub)
        steps.append(f"✓ Giảm độ sáng (Brightness -{beta_sub})")
        reasons.append(f"Độ sáng trung bình cao ({mean_val:.1f} > 175): Ảnh có nguy cơ cháy sáng.")

    # Bước 3: Khắc phục tương phản thấp
    if std_val < 42 or dyn_range < 130:
        alpha_val = float(np.clip(1.0 + (45 - std_val) / 60.0, 1.1, 1.45))
        result = adjust_contrast(result, alpha=alpha_val)
        result = apply_clahe(result, clip_limit=2.0, tile_grid_size=(8, 8))
        steps.append(f"✓ Tăng tương phản (Contrast x{alpha_val:.2f})")
        steps.append("✓ Áp dụng CLAHE (Clip Limit 2.0, Grid 8x8)")
        reasons.append(f"Độ tương phản thấp (Std: {std_val:.1f}, Dải động: {dyn_range:.0f}): Ảnh trông mờ nhạt.")

    # Bước 4: Kiểm tra và làm nét ảnh nếu ảnh thiếu sắc nét
    if laplacian_var < 150:
        result = sharpen_image(result, strength=35, method="Unsharp Mask")
        steps.append("✓ Làm nét ảnh (Sharpen Strength 35 - Unsharp Mask)")
        reasons.append(f"Độ sắc nét thấp (Laplacian Var: {laplacian_var:.1f} < 150): Các đường cạnh bị nhòe.")

    # Nếu ảnh đã ở trạng thái khá tối ưu, áp dụng CLAHE nhẹ nhàng để làm nổi bật chi tiết
    if not steps:
        result = apply_clahe(result, clip_limit=1.5, tile_grid_size=(8, 8))
        steps.append("✓ Tối ưu hóa vi tương phản (CLAHE nhẹ 1.5)")
        reasons.append("Ảnh có độ sáng và tương phản cân bằng, chỉ tăng cường nhẹ chi tiết vi mô.")

    return result, steps, reasons
