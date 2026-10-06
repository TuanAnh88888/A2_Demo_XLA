"""
Module: modules/histogram.py
Mục đích:
1. Tính toán Histogram cho ảnh Grayscale và ảnh màu (RGB).
2. Tính toán các chỉ số thống kê (Min, Max, Mean, Std, Median).
3. Tự động phân tích đặc điểm phân bố của ảnh và đưa ra nhận xét khoa học.
"""

import cv2
import numpy as np


def calculate_histogram(img_np, is_gray=True):
    """
    Tính toán histogram cho ảnh.
    Args:
        img_np (np.ndarray): Mảng ảnh uint8
        is_gray (bool): True nếu tính theo mức xám, False nếu tính theo RGB
    Returns:
        dict: Chứa các mảng histogram và trục bins (0-255).
    """
    bins = np.arange(256)
    
    if is_gray or img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
        # Nếu là ảnh màu mà yêu cầu tính xám thì chuyển đổi tạm thời
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            gray = img_np[:, :, 0]
        else:
            gray = img_np
            
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        return {
            "type": "gray",
            "bins": bins,
            "gray": hist
        }
    else:
        # Ảnh màu RGB (kênh 0: Red, kênh 1: Green, kênh 2: Blue)
        hist_r = cv2.calcHist([img_np], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_np], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_np], [2], None, [256], [0, 256]).flatten()
        
        # Cũng tính thêm kênh Grayscale tương đương để dễ tham khảo
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        return {
            "type": "rgb",
            "bins": bins,
            "red": hist_r,
            "green": hist_g,
            "blue": hist_b,
            "gray": hist_gray
        }


def calculate_cdf(hist):
    """
    Tính hàm phân bố tích lũy (Cumulative Distribution Function - CDF) từ Histogram.
    Chuẩn hóa về khoảng [0, 1].
    """
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    return cdf_normalized


def calculate_statistics(img_np):
    """
    Tính toán các giá trị thống kê định lượng của ảnh:
    - Min pixel
    - Max pixel
    - Mean (Độ sáng trung bình)
    - Std (Độ lệch chuẩn - phản ánh độ tương phản tổng thể)
    - Median (Trung vị)
    - Dynamic Range (Dải động: Max - Min)
    """
    if img_np is None:
        return {}

    # Nếu là ảnh màu, tính trên kênh độ sáng (Grayscale)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        gray = img_np[:, :, 0]
    else:
        gray = img_np

    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    median_val = float(np.median(gray))
    dynamic_range = max_val - min_val

    return {
        "min": min_val,
        "max": max_val,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "median": round(median_val, 2),
        "dynamic_range": dynamic_range
    }


def analyze_histogram(img_np):
    """
    Tự động phân tích ảnh và đưa ra nhận xét khoa học dựa trên số liệu thực tế.
    Trả về danh sách các nhận xét và gợi ý kỹ thuật xử lý phù hợp.
    """
    stats = calculate_statistics(img_np)
    if not stats:
        return {"summary": "Không có dữ liệu", "details": [], "recommendations": []}

    mean_val = stats["mean"]
    std_val = stats["std"]
    min_val = stats["min"]
    max_val = stats["max"]
    dynamic_range = stats["dynamic_range"]

    details = []
    recommendations = []

    # 1. Đánh giá độ sáng (Brightness)
    if mean_val < 80:
        details.append("Ảnh có xu hướng TỐI (Underexposed). Histogram tập trung lệch nhiều về phía bên trái (vùng mức xám thấp 0 - 85).")
        recommendations.append("Nên tăng độ sáng (Brightness) hoặc áp dụng Histogram Equalization / CLAHE để khôi phục chi tiết vùng bóng tối.")
    elif mean_val > 175:
        details.append("Ảnh có xu hướng SÁNG (Overexposed). Histogram tập trung lệch nhiều về phía bên phải (vùng mức xám cao 170 - 255).")
        recommendations.append("Nên giảm độ sáng (Brightness) hoặc giảm tương phản để tránh hiện tượng cháy sáng (clipping).")
    else:
        details.append(f"Ảnh có độ sáng trung bình ở mức cân bằng ({mean_val:.1f}/255). Mức sáng phân bố hài hòa giữa các vùng.")

    # 2. Đánh giá độ tương phản (Contrast)
    if std_val < 38 or dynamic_range < 120:
        details.append(f"Độ tương phản THẤP (Low Contrast). Độ lệch chuẩn đạt {std_val:.1f}, dải mức xám hẹp ({dynamic_range:.0f} mức xám). Ảnh trông mờ nhạt và kém rõ nét.")
        recommendations.append("Áp dụng Histogram Equalization hoặc tăng Contrast để kéo giãn dải động ra toàn dải [0, 255].")
    elif std_val >= 55 and dynamic_range >= 190:
        details.append(f"Độ tương phản CAO (High Contrast). Độ lệch chuẩn đạt {std_val:.1f}, dải mức xám rộng ({dynamic_range:.0f} mức xám). Sự phân biệt giữa vùng sáng và vùng tối rất rõ rệt.")
    else:
        details.append(f"Độ tương phản ở mức TRUNG BÌNH (Medium Contrast - Std: {std_val:.1f}). Chi tiết thể hiện tương đối tốt.")

    # 3. Đánh giá bão hòa hoặc suy giảm chi tiết
    if min_val > 30 and max_val < 225:
        details.append("Dải mức xám bị co cụm ở giữa, không tận dụng hết khoảng giá trị hiển thị [0, 255].")
        recommendations.append("Phương pháp Histogram Equalization sẽ rất hiệu quả với bức ảnh này.")

    # 4. Gợi ý kỹ thuật nâng cao
    if std_val < 45 or mean_val < 90 or mean_val > 165:
        recommendations.append("Nếu ảnh có vùng chiếu sáng không đều (vừa có vùng quá sáng vừa có vùng quá tối), kỹ thuật CLAHE là giải pháp tối ưu nhất để tránh khuếch đại nhiễu.")

    return {
        "stats": stats,
        "details": details,
        "recommendations": recommendations
    }
