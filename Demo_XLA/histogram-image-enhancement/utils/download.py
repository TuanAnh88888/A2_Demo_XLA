"""
Module: utils/download.py
Mục đích:
1. Xuất và đóng gói dữ liệu ảnh để tải về qua giao diện Streamlit.
2. Hỗ trợ định dạng PNG (nén không mất dữ liệu) và JPEG/JPG (dung lượng nhỏ gọn).
3. Đặt tên file thông minh theo từng giai đoạn xử lý.
"""

import io
from PIL import Image
import numpy as np


def prepare_image_download(img_np, filename="result", file_format="PNG", quality=95):
    """
    Chuẩn bị dữ liệu bytes và mime-type tương ứng để tải về.
    
    Args:
        img_np (np.ndarray): Mảng ảnh uint8
        filename (str): Tên file cơ sở
        file_format (str): 'PNG' hoặc 'JPEG'/'JPG'
        quality (int): Chất lượng cho ảnh JPEG (1-100)
    Returns:
        tuple: (bytes_data, file_name, mime_type)
    """
    if img_np is None:
        return b"", "empty.png", "image/png"

    # Tạo đối tượng PIL Image
    if img_np.ndim == 2:
        pil_img = Image.fromarray(img_np, mode="L")
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        pil_img = Image.fromarray(img_np[:, :, 0], mode="L")
    else:
        pil_img = Image.fromarray(img_np, mode="RGB")

    buf = io.BytesIO()
    fmt_upper = file_format.upper()
    if fmt_upper in ["JPG", "JPEG"]:
        pil_img.save(buf, format="JPEG", quality=quality)
        ext = "jpg"
        mime = "image/jpeg"
    else:
        pil_img.save(buf, format="PNG")
        ext = "png"
        mime = "image/png"

    clean_name = filename.rsplit(".", 1)[0]
    out_name = f"{clean_name}.{ext}"
    return buf.getvalue(), out_name, mime
