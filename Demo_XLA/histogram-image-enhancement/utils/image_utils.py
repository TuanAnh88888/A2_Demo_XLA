"""
Module: utils/image_utils.py
Mục đích: Cung cấp các hàm tiện ích đọc, chuyển đổi và xử lý ảnh cơ bản.
Phù hợp cho cả ảnh Grayscale và ảnh Màu (RGB).
"""

import io
import os
import cv2
import numpy as np
from PIL import Image


def load_image(image_input):
    """
    Đọc ảnh từ file upload Streamlit (UploadedFile), đường dẫn file (str), hoặc bytes.
    Trả về:
        numpy.ndarray: Ảnh ở định dạng RGB hoặc Grayscale (uint8), hoặc None nếu lỗi.
        str: Thông báo lỗi nếu có.
    """
    try:
        if image_input is None:
            return None, "Không có dữ liệu ảnh được cung cấp."

        # Trường hợp 1: Streamlit UploadedFile hoặc BytesIO
        if hasattr(image_input, "read"):
            # Đảm bảo con trỏ file ở đầu
            image_input.seek(0)
            file_bytes = image_input.read()
            # Mở bằng PIL để đảm bảo tương thích mọi định dạng (PNG, JPG, BMP, WEBP,...)
            pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Trường hợp 2: Đường dẫn file cục bộ (chuỗi)
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None, f"Tệp tin không tồn tại: {image_input}"
            pil_img = Image.open(image_input)
            
        # Trường hợp 3: Đối tượng PIL Image
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            
        # Trường hợp 4: Đối tượng numpy array đã có sẵn
        elif isinstance(image_input, np.ndarray):
            return image_input.copy(), None
            
        else:
            return None, "Định dạng đầu vào không được hỗ trợ."

        # Xử lý các mode màu của PIL
        if pil_img.mode == "RGBA":
            # Chuyển RGBA sang RGB nền trắng để tránh lỗi kênh alpha
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.split()[3])
            img_np = np.array(background, dtype=np.uint8)
        elif pil_img.mode == "L":
            # Ảnh xám 1 kênh
            img_np = np.array(pil_img, dtype=np.uint8)
        else:
            # Chuyển sang RGB chuẩn
            rgb_img = pil_img.convert("RGB")
            img_np = np.array(rgb_img, dtype=np.uint8)

        return img_np, None

    except Exception as e:
        return None, f"Không thể đọc ảnh. Lỗi chi tiết: {str(e)}"


def get_image_info(img_np, filename="Ảnh tải lên"):
    """
    Lấy thông tin chi tiết về ảnh: kích thước, số kênh màu, kiểu dữ liệu, định dạng.
    """
    if img_np is None:
        return {}

    height, width = img_np.shape[:2]
    channels = 1 if img_np.ndim == 2 else img_np.shape[2]
    color_type = "Ảnh Grayscale (Mức xám)" if channels == 1 else "Ảnh Màu (RGB)"

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "size_str": f"{width} × {height} pixels",
        "channels": channels,
        "color_type": color_type,
        "dtype": str(img_np.dtype),
        "total_pixels": width * height,
        "memory_kb": round((img_np.nbytes) / 1024, 2)
    }


def is_grayscale(img_np):
    """
    Kiểm tra xem ảnh có phải ảnh đơn sắc/mức xám hay không.
    """
    if img_np is None:
        return False
    if img_np.ndim == 2:
        return True
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return True
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        # Nếu cả 3 kênh R, G, B bằng nhau hoàn toàn thì thực chất là ảnh xám
        return bool(np.array_equal(img_np[:, :, 0], img_np[:, :, 1]) and np.array_equal(img_np[:, :, 1], img_np[:, :, 2]))
    return False


def convert_to_gray(img_np):
    """
    Chuyển ảnh RGB sang ảnh mức xám 2D (uint8). Nếu đã là ảnh xám thì giữ nguyên.
    """
    if img_np is None:
        return None
    if img_np.ndim == 2:
        return img_np.copy()
    if img_np.ndim == 3:
        if img_np.shape[2] == 1:
            return img_np[:, :, 0].copy()
        elif img_np.shape[2] == 3:
            # OpenCV quy ước RGB -> GRAY
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.shape[2] == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    return img_np


def ensure_rgb(img_np):
    """
    Đảm bảo ảnh có 3 kênh (RGB) để hiển thị đồng nhất trên web.
    Nếu là ảnh xám 2D, chuyển thành 3 kênh giống nhau.
    """
    if img_np is None:
        return None
    if img_np.ndim == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        return cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2RGB)
    return img_np


def convert_to_bytes(img_np, format_type="PNG"):
    """
    Chuyển đổi mảng numpy thành mảng bytes để phục vụ tải về qua st.download_button.
    """
    if img_np is None:
        return b""
    
    # Chuẩn bị ảnh PIL để lưu sang bytes
    if img_np.ndim == 2:
        pil_img = Image.fromarray(img_np, mode="L")
    else:
        pil_img = Image.fromarray(img_np, mode="RGB")
        
    buf = io.BytesIO()
    pil_img.save(buf, format=format_type)
    return buf.getvalue()
