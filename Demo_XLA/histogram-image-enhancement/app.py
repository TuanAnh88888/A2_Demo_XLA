"""
=============================================================================
DỰ ÁN: HISTOGRAM VÀ TĂNG CƯỜNG CHẤT LƯỢNG ẢNH
Hệ thống xử lý và tăng cường chất lượng ảnh số tuần hoàn & liên tục
Môi trường phát triển: Visual Studio Code (Windows)
Ngôn ngữ: Python 3.x | Giao diện: Streamlit
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Đảm bảo đường dẫn import tương thích tuyệt đối
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.image_utils import (
    load_image,
    get_image_info,
    convert_to_gray,
    convert_to_bytes,
    is_grayscale,
    ensure_rgb,
)
from utils.visualization import (
    plot_histogram_matplotlib,
    plot_comparison_histograms,
    plot_evaluation_metrics_bar,
    plot_multi_histogram,
)
from utils.history import (
    init_image_state,
    apply_step,
    undo_step,
    redo_step,
    reset_to_original,
)
from utils.download import prepare_image_download

from modules.histogram import (
    calculate_histogram,
    calculate_statistics,
    analyze_histogram,
    calculate_cdf,
)
from modules.equalization import (
    equalize_histogram_opencv,
    equalize_histogram_numpy,
    equalize_histogram_rgb_direct,
)
from modules.clahe import apply_clahe
from modules.brightness import adjust_brightness
from modules.contrast import adjust_contrast, adjust_brightness_contrast
from modules.gamma import adjust_gamma, adjust_exposure, adjust_highlights_shadows
from modules.sharpening import sharpen_image, enhance_details, enhance_edges
from modules.denoising import apply_denoising
from modules.deblurring import deblur_image
from modules.color import (
    adjust_saturation,
    adjust_hue,
    adjust_temperature,
    apply_sepia,
    convert_to_grayscale,
)
from modules.white_balance import apply_white_balance
from modules.auto_enhance import auto_enhance_image
from modules.evaluation import (
    evaluate_all,
    calculate_mean,
    calculate_std,
    calculate_entropy,
    calculate_psnr,
    calculate_ssim,
)

# =============================================================================
# CẤU HÌNH TRANG STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Histogram & Image Enhancement System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS giao diện chuyên nghiệp, hiện đại, tối ưu hiển thị
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .history-badge {
        display: inline-block;
        padding: 3px 8px;
        background-color: #E0E7FF;
        color: #3730A3;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .status-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .preview-box {
        border: 2px dashed #3B82F6;
        border-radius: 8px;
        padding: 8px;
        background-color: #EFF6FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tiêu đề & phụ đề chính
st.markdown('<div class="main-title">HISTOGRAM VÀ TĂNG CƯỜNG CHẤT LƯỢNG ẢNH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Histogram – Histogram Equalization – CLAHE – Image Enhancement System</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# KHỞI TẠO SESSION STATE QUẢN LÝ ẢNH
# =============================================================================
if "original_image" not in st.session_state:
    st.session_state["original_image"] = None
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None
if "preview_image" not in st.session_state:
    st.session_state["preview_image"] = None
if "preview_name" not in st.session_state:
    st.session_state["preview_name"] = ""
if "image_name" not in st.session_state:
    st.session_state["image_name"] = "Chưa có ảnh"
if "history" not in st.session_state:
    st.session_state["history"] = []
if "undo_stack" not in st.session_state:
    st.session_state["undo_stack"] = []
if "redo_stack" not in st.session_state:
    st.session_state["redo_stack"] = []

# =============================================================================
# SIDEBAR: MENU & UPLOAD
# =============================================================================
st.sidebar.title("📌 MENU ĐIỀU HƯỚNG")

menu_choice = st.sidebar.radio(
    "Chọn phân hệ chức năng:",
    [
        "1. Trang chủ",
        "2. Xử lý & cải thiện ảnh",
        "3. Histogram & phân tích",
        "4. So sánh thuật toán",
        "5. Đánh giá kết quả",
        "6. So sánh Before / After",
        "7. Demo tự động & Thực nghiệm",
        "8. Cơ sở lý thuyết",
    ],
    index=1 if st.session_state.get("current_image") is not None else 0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📤 TẢI LÊN ẢNH XỬ LÝ")

uploaded_file = st.sidebar.file_uploader(
    "Chọn ảnh từ máy tính:",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Hỗ trợ đầy đủ các định dạng: .jpg, .jpeg, .png, .bmp, .webp",
)

sample_dir = os.path.join(CURRENT_DIR, "sample_images")
sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(sample_dir) else []

selected_sample = None
if sample_files:
    sample_choice = st.sidebar.selectbox(
        "Hoặc chọn ảnh mẫu thử nghiệm:",
        ["-- Không dùng ảnh mẫu --"] + sample_files,
        index=0,
        help="Thử nghiệm nhanh các trường hợp điển hình: thiếu sáng, tương phản thấp, chiếu sáng không đều...",
    )
    if sample_choice != "-- Không dùng ảnh mẫu --":
        selected_sample = os.path.join(sample_dir, sample_choice)

# Xử lý cập nhật ảnh mới khi người dùng tải lên hoặc chọn mẫu
new_image_to_load = None
loaded_name = ""

if uploaded_file is not None:
    # Kiểm tra xem có phải file mới tải lên so với session hiện tại
    if uploaded_file.name != st.session_state.get("uploaded_filename"):
        img_np, err = load_image(uploaded_file)
        if err:
            st.sidebar.error(f"Lỗi đọc ảnh: {err}")
        else:
            new_image_to_load = img_np
            loaded_name = uploaded_file.name
            st.session_state["uploaded_filename"] = uploaded_file.name
elif selected_sample is not None:
    if selected_sample != st.session_state.get("last_sample_path"):
        img_np, err = load_image(selected_sample)
        if err:
            st.sidebar.error(f"Lỗi: {err}")
        else:
            new_image_to_load = img_np
            loaded_name = os.path.basename(selected_sample)
            st.session_state["last_sample_path"] = selected_sample

if new_image_to_load is not None:
    init_image_state(st.session_state, new_image_to_load, loaded_name)
    st.sidebar.success(f"Đã nạp thành công ảnh: {loaded_name}")

# Hiển thị metadata ảnh trên Sidebar nếu đã có ảnh
if st.session_state.get("original_image") is not None:
    info = get_image_info(st.session_state["current_image"], st.session_state["image_name"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ THÔNG TIN ẢNH")
    st.sidebar.markdown(f"**Tên file:** `{info['filename']}`")
    st.sidebar.markdown(f"**Kích thước:** `{info['size_str']}`")
    st.sidebar.markdown(f"**Số kênh:** `{info['channels']}` ({info['color_type']})")
    st.sidebar.markdown(f"**Định dạng:** `{info['dtype'].upper()}`")
    st.sidebar.markdown(f"**Số bước đã chỉnh sửa:** `{len(st.session_state['history']) - 1}`")


def require_image():
    """Kiểm tra sự tồn tại của ảnh trong session_state"""
    if st.session_state.get("current_image") is None:
        st.warning("⚠️ **Vui lòng upload ảnh để bắt đầu.**")
        st.info("💡 Bạn có thể chọn ảnh từ máy tính ở thanh bên trái (Sidebar) hoặc chọn một trong các **ảnh mẫu có sẵn**.")
        return False
    return True


# =============================================================================
# 1. TRANG CHỦ
# =============================================================================
if menu_choice == "1. Trang chủ":
    st.subheader("Chào mừng đến với Hệ thống Phân tích & Tăng cường Chất lượng Ảnh")
    
    col_intro, col_side = st.columns([1.2, 0.8])
    with col_intro:
        st.markdown(
            """
            ### 📖 Giới thiệu
            **Histogram** là công cụ biểu diễn sự phân bố mức cường độ sáng của các pixel trong ảnh số.
            Ứng dụng sử dụng Histogram để phân tích ảnh và kết hợp nhiều kỹ thuật tiên tiến để tăng cường chất lượng ảnh.

            Khác với các ứng dụng minh họa rời rạc, hệ thống này hoạt động như một **quy trình xử lý ảnh chuyên nghiệp**:
            - 🔹 **Bảo toàn ảnh gốc (`original_image`):** Không bao giờ bị ghi đè.
            - 🔹 **Chỉnh sửa liên tiếp (`current_image`):** Mọi thao tác kế thừa kết quả của bước trước đó.
            - 🔹 **Kiểm soát linh hoạt:** Hỗ trợ đầy đủ **Xem trước (Preview)**, **Áp dụng (Apply)**, **Hoàn tác (Undo)**, **Làm lại (Redo)** và **Lịch sử chỉnh sửa**.
            """
        )

        st.markdown(
            """
            ### 🛠️ Các nhóm chức năng cốt lõi:
            - **Phân tích Histogram:** Phân tích mức xám, từng kênh RGB, CDF và tự động chẩn đoán.
            - **Điều chỉnh ánh sáng & tương phản:** Brightness, Contrast, Gamma Correction, Exposure, Highlights & Shadows.
            - **Cân bằng lược đồ:** Histogram Equalization toàn cục (YCrCb bảo toàn màu) và CLAHE cục bộ.
            - **Khử nhiễu (Denoising):** Gaussian Blur, Median, Bilateral Filter, Non-Local Means.
            - **Làm nét & Khôi phục (Sharpening & Deblur):** Unsharp Mask, High-pass, Richardson-Lucy, Wiener.
            - **Màu sắc & Cân bằng trắng:** Saturation, Hue, Temperature, Gray World White Balance.
            - **Tự động cải thiện (Auto Enhance):** Chẩn đoán và kích hoạt chuỗi nâng cấp tối ưu.
            - **Đánh giá & So sánh:** Đối chiếu định lượng Mean, Std, Entropy, PSNR, SSIM và đo thời gian xử lý.
            """
        )

    with col_side:
        if st.session_state.get("current_image") is not None:
            st.image(st.session_state["current_image"], caption=f"Ảnh hiện tại ({st.session_state['image_name']})", use_container_width=True)
            stats = calculate_statistics(st.session_state["current_image"])
            st.markdown(
                f"""
                <div class="status-panel">
                <b>Trạng thái ảnh hiện tại:</b><br>
                • Độ sáng TB (Mean): <b>{stats['mean']}</b> / 255<br>
                • Độ tương phản (Std): <b>{stats['std']}</b><br>
                • Lịch sử: <b>{len(st.session_state['history'])} bước</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Hãy tải ảnh lên từ thanh bên trái (Sidebar) để trải nghiệm hệ thống!")


# =============================================================================
# 2. XỬ LÝ & CẢI THIỆN ẢNH (TRUNG TÂM CỦA HỆ THỐNG)
# =============================================================================
elif menu_choice == "2. Xử lý & cải thiện ảnh":
    st.header("🛠️ Trung Tâm Xử Lý & Cải Thiện Ảnh (Chỉnh Sửa Liên Tục)")
    
    if require_image():
        orig_img = st.session_state["original_image"]
        curr_img = st.session_state["current_image"]
        preview_img = st.session_state.get("preview_image")

        # -------------------------------------------------------------
        # KHU VỰC HIỂN THỊ: ẢNH GỐC & ẢNH ĐANG CHỈNH SỬA
        # -------------------------------------------------------------
        c_orig, c_curr = st.columns(2)
        with c_orig:
            st.markdown("### 🖼️ ẢNH GỐC (ORIGINAL)")
            st.image(orig_img, caption="Ảnh gốc ban đầu (Không đổi)", use_container_width=True)

        with c_curr:
            if preview_img is not None:
                st.markdown(f"### 👁️ ĐANG XEM TRƯỚC: `{st.session_state.get('preview_name', '')}`")
                st.image(preview_img, caption="Bản xem trước tạm thời (Nhấn 'Áp dụng' để lưu)", use_container_width=True)
            else:
                last_op = st.session_state["history"][-1]["name"] if st.session_state.get("history") else "Original"
                st.markdown(f"### ✨ ẢNH HIỆN TẠI (CURRENT) — Bước: `{last_op}`")
                st.image(curr_img, caption=f"Kế thừa từ {len(st.session_state['history']) - 1} bước xử lý trước", use_container_width=True)

        # -------------------------------------------------------------
        # THANH CÔNG CỤ ĐIỀU KHIỂN: UNDO / REDO / RESET / DOWNLOAD
        # -------------------------------------------------------------
        btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns([1, 1, 1.2, 1.2, 1.2])

        with btn_c1:
            can_undo = len(st.session_state.get("undo_stack", [])) > 0
            if st.button("↩ Hoàn tác", disabled=not can_undo, use_container_width=True, help="Quay lại bước trước đó"):
                if undo_step(st.session_state):
                    st.rerun()

        with btn_c2:
            can_redo = len(st.session_state.get("redo_stack", [])) > 0
            if st.button("↪ Làm lại", disabled=not can_redo, use_container_width=True, help="Khôi phục bước vừa hoàn tác"):
                if redo_step(st.session_state):
                    st.rerun()

        with btn_c3:
            if preview_img is not None:
                if st.button("❌ Hủy xem trước", use_container_width=True):
                    st.session_state["preview_image"] = None
                    st.session_state["preview_name"] = ""
                    st.rerun()
            else:
                if st.button("↻ Khôi phục gốc", type="secondary", use_container_width=True, help="Đặt lại ảnh về nguyên bản ban đầu"):
                    reset_to_original(st.session_state)
                    st.success("Đã khôi phục về ảnh gốc ban đầu!")
                    st.rerun()

        with btn_c4:
            # Tải ảnh hiện tại
            b_data, b_name, b_mime = prepare_image_download(curr_img, f"enhanced_{st.session_state['image_name']}", "PNG")
            st.download_button(
                label="📥 Tải ảnh hiện tại (PNG)",
                data=b_data,
                file_name=b_name,
                mime=b_mime,
                use_container_width=True,
            )

        with btn_c5:
            # Tải định dạng JPG
            j_data, j_name, j_mime = prepare_image_download(curr_img, f"enhanced_{st.session_state['image_name']}", "JPG")
            st.download_button(
                label="📥 Tải ảnh hiện tại (JPG)",
                data=j_data,
                file_name=j_name,
                mime=j_mime,
                use_container_width=True,
            )

        st.markdown("---")

        # -------------------------------------------------------------
        # CÁC NHÓM CÔNG CỤ CHỈNH SỬA TUẦN TỰ (TÁC ĐỘNG LÊN CURRENT_IMAGE)
        # -------------------------------------------------------------
        tab_light, tab_hist, tab_denoise, tab_sharp, tab_color, tab_auto, tab_hist_log = st.tabs(
            [
                "☀️ 1. Ánh sáng & Tương phản",
                "⚖️ 2. Histogram Equalization & CLAHE",
                "🧹 3. Khử nhiễu",
                "🔍 4. Làm nét & Chi tiết",
                "🎨 5. Màu sắc & Cân bằng trắng",
                "🤖 6. Tự động cải thiện (Auto)",
                "📜 7. Lịch sử chỉnh sửa",
            ]
        )

        # TAB 1: ÁNH SÁNG & TƯƠNG PHẢN
        with tab_light:
            st.markdown("##### Điều chỉnh thông số ánh sáng tác động lên `current_image`")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                b_val = st.slider("Độ sáng (Brightness - β):", min_value=-100, max_value=100, value=0, step=1)
                c_val = st.slider("Độ tương phản (Contrast - α):", min_value=0.1, max_value=3.0, value=1.0, step=0.05)
                g_val = st.slider("Gamma Correction (γ):", min_value=0.2, max_value=2.8, value=1.0, step=0.05)
            with col_l2:
                exp_val = st.slider("Mức phơi sáng (Exposure EV):", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
                hl_val = st.slider("Vùng sáng (Highlights):", min_value=-100, max_value=100, value=0, step=5)
                sd_val = st.slider("Vùng tối (Shadows):", min_value=-100, max_value=100, value=0, step=5)

            btn_prev_l, btn_apply_l = st.columns([1, 1])
            with btn_prev_l:
                if st.button("👁️ Xem trước Ánh sáng", use_container_width=True):
                    res = curr_img.copy()
                    if b_val != 0:
                        res = adjust_brightness(res, beta=b_val)
                    if c_val != 1.0:
                        res = adjust_contrast(res, alpha=c_val)
                    if g_val != 1.0:
                        res = adjust_gamma(res, gamma=g_val)
                    if exp_val != 0.0:
                        res = adjust_exposure(res, ev=exp_val)
                    if hl_val != 0 or sd_val != 0:
                        res = adjust_highlights_shadows(res, highlights=hl_val, shadows=sd_val)
                    st.session_state["preview_image"] = res
                    st.session_state["preview_name"] = f"Light (B:{b_val:+d}, C:{c_val:.2f}, γ:{g_val:.2f})"
                    st.rerun()

            with btn_apply_l:
                if st.button("✅ ÁP DỤNG ÁNH SÁNG", type="primary", use_container_width=True):
                    res = curr_img.copy()
                    desc_parts = []
                    if b_val != 0:
                        res = adjust_brightness(res, beta=b_val)
                        desc_parts.append(f"B{b_val:+d}")
                    if c_val != 1.0:
                        res = adjust_contrast(res, alpha=c_val)
                        desc_parts.append(f"C{c_val:.2f}")
                    if g_val != 1.0:
                        res = adjust_gamma(res, gamma=g_val)
                        desc_parts.append(f"γ{g_val:.2f}")
                    if exp_val != 0.0:
                        res = adjust_exposure(res, ev=exp_val)
                        desc_parts.append(f"EV{exp_val:+.1f}")
                    if hl_val != 0 or sd_val != 0:
                        res = adjust_highlights_shadows(res, highlights=hl_val, shadows=sd_val)
                        desc_parts.append(f"HL{hl_val:+d}/SD{sd_val:+d}")
                    
                    op_name = f"Ánh sáng ({', '.join(desc_parts)})" if desc_parts else "Ánh sáng (Giữ nguyên)"
                    apply_step(st.session_state, res, op_name, {"b": b_val, "c": c_val, "g": g_val})
                    st.success(f"Đã áp dụng: {op_name}")
                    st.rerun()

        # TAB 2: HISTOGRAM EQUALIZATION & CLAHE
        with tab_hist:
            st.markdown("##### Cân bằng lược đồ mức xám tác động lên `current_image`")
            he_col, clahe_col = st.columns(2)

            with he_col:
                st.markdown("###### 1. Histogram Equalization (Toàn cục)")
                he_mode = st.selectbox(
                    "Phương thức cân bằng:",
                    ["OpenCV (Chuẩn YCrCb - Bảo toàn màu)", "NumPy CDF (Giải tích lý thuyết)", "RGB Direct (Minh họa lệch màu)"],
                )
                if st.button("👁️ Xem trước Equalization", use_container_width=True):
                    if "OpenCV" in he_mode:
                        st.session_state["preview_image"] = equalize_histogram_opencv(curr_img)
                    elif "NumPy" in he_mode:
                        st.session_state["preview_image"], _, _ = equalize_histogram_numpy(curr_img)
                    else:
                        st.session_state["preview_image"] = equalize_histogram_rgb_direct(curr_img)
                    st.session_state["preview_name"] = "Histogram Equalization"
                    st.rerun()

                if st.button("✅ ÁP DỤNG EQUALIZATION", type="primary", use_container_width=True):
                    if "OpenCV" in he_mode:
                        res = equalize_histogram_opencv(curr_img)
                    elif "NumPy" in he_mode:
                        res, _, _ = equalize_histogram_numpy(curr_img)
                    else:
                        res = equalize_histogram_rgb_direct(curr_img)
                    apply_step(st.session_state, res, "Histogram Equalization", {"mode": he_mode})
                    st.success("Đã áp dụng Histogram Equalization lên ảnh hiện tại!")
                    st.rerun()

            with clahe_col:
                st.markdown("###### 2. CLAHE (Cục bộ & Giới hạn tương phản)")
                c_clip = st.slider("Clip Limit (Giới hạn cắt):", 0.5, 5.0, 2.0, 0.1)
                c_grid = st.select_slider("Tile Grid Size:", options=["2x2", "4x4", "8x8", "16x16"], value="8x8")
                grid_m = {"2x2": (2, 2), "4x4": (4, 4), "8x8": (8, 8), "16x16": (16, 16)}

                if st.button("👁️ Xem trước CLAHE", use_container_width=True):
                    st.session_state["preview_image"] = apply_clahe(curr_img, clip_limit=c_clip, tile_grid_size=grid_m[c_grid])
                    st.session_state["preview_name"] = f"CLAHE (Clip:{c_clip}, Grid:{c_grid})"
                    st.rerun()

                if st.button("✅ ÁP DỤNG CLAHE", type="primary", use_container_width=True):
                    res = apply_clahe(curr_img, clip_limit=c_clip, tile_grid_size=grid_m[c_grid])
                    apply_step(st.session_state, res, f"CLAHE (Clip {c_clip}, Grid {c_grid})", {"clip": c_clip, "grid": c_grid})
                    st.success(f"Đã áp dụng CLAHE (Clip={c_clip}, Grid={c_grid})!")
                    st.rerun()

        # TAB 3: KHỬ NHIỄU (DENOISING)
        with tab_denoise:
            st.markdown("##### Bộ lọc khử nhiễu tác động lên `current_image`")
            dn_col1, dn_col2 = st.columns(2)
            with dn_col1:
                dn_method = st.selectbox(
                    "Thuật toán khử nhiễu:",
                    ["Bilateral Filter (Bảo toàn cạnh sắc nét)", "Gaussian Blur (Làm mịn Gauss)", "Median Filter (Khử nhiễu muối tiêu)", "Non-Local Means (Chất lượng cao)"],
                )
            with dn_col2:
                dn_level = st.radio("Cấp độ khử nhiễu:", ["Nhẹ", "Vừa", "Mạnh"], horizontal=True, index=1)

            btn_prev_dn, btn_apply_dn = st.columns(2)
            clean_dn_name = dn_method.split(" (")[0]
            with btn_prev_dn:
                if st.button("👁️ Xem trước Khử nhiễu", use_container_width=True):
                    st.session_state["preview_image"] = apply_denoising(curr_img, method=clean_dn_name, level=dn_level)
                    st.session_state["preview_name"] = f"Khử nhiễu: {clean_dn_name} ({dn_level})"
                    st.rerun()

            with btn_apply_dn:
                if st.button("✅ ÁP DỤNG KHỬ NHIỄU", type="primary", use_container_width=True):
                    res = apply_denoising(curr_img, method=clean_dn_name, level=dn_level)
                    apply_step(st.session_state, res, f"Khử nhiễu ({clean_dn_name} - {dn_level})", {"method": clean_dn_name, "level": dn_level})
                    st.success(f"Đã khử nhiễu bằng {clean_dn_name} ({dn_level})!")
                    st.rerun()

        # TAB 4: LÀM NÉT & CHI TIẾT (SHARPENING & DEBLUR)
        with tab_sharp:
            st.markdown("##### Nâng cao độ sắc nét và phục hồi độ mờ cho `current_image`")
            sh_c1, sh_c2 = st.columns(2)

            with sh_c1:
                st.markdown("###### 1. Làm nét ảnh (Sharpening)")
                sh_strength = st.slider("Cường độ nét (Strength):", 0, 100, 40, 5)
                sh_method = st.selectbox("Phương pháp làm nét:", ["Unsharp Mask", "Kernel Sharpen", "High-Pass"])
                
                sp1, sp2 = st.columns(2)
                with sp1:
                    if st.button("👁️ Xem trước Làm nét", use_container_width=True):
                        st.session_state["preview_image"] = sharpen_image(curr_img, strength=sh_strength, method=sh_method)
                        st.session_state["preview_name"] = f"Sharpen ({sh_method} - {sh_strength})"
                        st.rerun()
                with sp2:
                    if st.button("✅ ÁP DỤNG LÀM NÉT", type="primary", use_container_width=True):
                        res = sharpen_image(curr_img, strength=sh_strength, method=sh_method)
                        apply_step(st.session_state, res, f"Làm nét ({sh_method} +{sh_strength})", {"strength": sh_strength, "method": sh_method})
                        st.success("Đã làm nét ảnh thành công!")
                        st.rerun()

            with sh_c2:
                st.markdown("###### 2. Cải thiện ảnh mờ & Tăng cường chi tiết")
                deblur_opt = st.selectbox(
                    "Phương thức phục hồi:",
                    ["Tăng chi tiết bề mặt (Detail Enhance)", "Tăng cường đường biên (Edge Enhance)", "Mờ nhẹ (Unsharp Deblur)", "Mờ Gaussian (Richardson-Lucy)", "Mờ chuyển động (Wiener)"],
                )
                dp1, dp2 = st.columns(2)
                with dp1:
                    if st.button("👁️ Xem trước Chi tiết/Mờ", use_container_width=True):
                        if "Detail" in deblur_opt:
                            st.session_state["preview_image"] = enhance_details(curr_img, strength=50)
                        elif "Edge" in deblur_opt:
                            st.session_state["preview_image"] = enhance_edges(curr_img, strength=50)
                        else:
                            clean_deb = deblur_opt.split(" (")[0]
                            st.session_state["preview_image"], _ = deblur_image(curr_img, blur_type=clean_deb)
                        st.session_state["preview_name"] = deblur_opt
                        st.rerun()
                with dp2:
                    if st.button("✅ ÁP DỤNG PHỤC HỒI", type="primary", use_container_width=True):
                        if "Detail" in deblur_opt:
                            res = enhance_details(curr_img, strength=50)
                        elif "Edge" in deblur_opt:
                            res = enhance_edges(curr_img, strength=50)
                        else:
                            clean_deb = deblur_opt.split(" (")[0]
                            res, _ = deblur_image(curr_img, blur_type=clean_deb)
                        apply_step(st.session_state, res, deblur_opt, {})
                        st.success(f"Đã áp dụng: {deblur_opt}!")
                        st.rerun()

        # TAB 5: MÀU SẮC & CÂN BẰNG TRẮNG
        with tab_color:
            st.markdown("##### Tinh chỉnh sắc độ và nhiệt độ màu cho `current_image`")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                sat_val = st.slider("Độ bão hòa màu (Saturation %):", 0, 200, 100, 5)
                hue_val = st.slider("Sắc thái màu (Hue shift):", -90, 90, 0, 2)
                temp_val = st.slider("Nhiệt độ màu (Temperature: Lạnh <-> Ấm):", -100, 100, 0, 5)
            with col_m2:
                wb_mode = st.selectbox("Cân bằng trắng (White Balance):", ["Không dùng", "Gray World (Thế giới xám)", "White Patch (Điểm trắng chuẩn)"])
                filter_mode = st.selectbox("Hiệu ứng màu đặc biệt:", ["Không dùng", "Chuyển sang Grayscale (Mức xám)", "Hiệu ứng Sepia cổ điển"])

            btn_prev_c, btn_apply_c = st.columns(2)
            with btn_prev_c:
                if st.button("👁️ Xem trước Màu sắc", use_container_width=True):
                    res = curr_img.copy()
                    if sat_val != 100:
                        res = adjust_saturation(res, saturation_percent=sat_val)
                    if hue_val != 0:
                        res = adjust_hue(res, hue_shift=hue_val)
                    if temp_val != 0:
                        res = adjust_temperature(res, temp_val=temp_val)
                    if "Gray World" in wb_mode:
                        res = apply_white_balance(res, method="Gray World")
                    elif "White Patch" in wb_mode:
                        res = apply_white_balance(res, method="White Patch")
                    if "Grayscale" in filter_mode:
                        res = convert_to_grayscale(res)
                    elif "Sepia" in filter_mode:
                        res = apply_sepia(res)
                    st.session_state["preview_image"] = res
                    st.session_state["preview_name"] = f"Color (Sat:{sat_val}%, Temp:{temp_val})"
                    st.rerun()

            with btn_apply_c:
                if st.button("✅ ÁP DỤNG MÀU SẮC", type="primary", use_container_width=True):
                    res = curr_img.copy()
                    desc_col = []
                    if sat_val != 100:
                        res = adjust_saturation(res, saturation_percent=sat_val)
                        desc_col.append(f"Sat {sat_val}%")
                    if hue_val != 0:
                        res = adjust_hue(res, hue_shift=hue_val)
                        desc_col.append(f"Hue {hue_val}")
                    if temp_val != 0:
                        res = adjust_temperature(res, temp_val=temp_val)
                        desc_col.append(f"Temp {temp_val}")
                    if "Gray World" in wb_mode:
                        res = apply_white_balance(res, method="Gray World")
                        desc_col.append("WB GrayWorld")
                    elif "White Patch" in wb_mode:
                        res = apply_white_balance(res, method="White Patch")
                        desc_col.append("WB WhitePatch")
                    if "Grayscale" in filter_mode:
                        res = convert_to_grayscale(res)
                        desc_col.append("Grayscale")
                    elif "Sepia" in filter_mode:
                        res = apply_sepia(res)
                        desc_col.append("Sepia")

                    op_title = f"Màu sắc ({', '.join(desc_col)})" if desc_col else "Màu sắc (Giữ nguyên)"
                    apply_step(st.session_state, res, op_title, {})
                    st.success(f"Đã cập nhật: {op_title}!")
                    st.rerun()

        # TAB 6: TỰ ĐỘNG CẢI THIỆN (AUTO ENHANCE)
        with tab_auto:
            st.markdown("##### Hệ thống tự động phân tích `current_image` và chạy chuỗi tối ưu thích nghi")
            st.info("Hệ thống sẽ không chạy máy móc tất cả các thuật toán, mà kiểm tra số liệu thực tế (độ sáng, tương phản, độ sắc nét, ám màu) để đưa ra phác đồ tăng cường phù hợp nhất.")

            if st.button("🚀 CHẠY TỰ ĐỘNG CẢI THIỆN", type="primary", use_container_width=True):
                with st.spinner("Đang chẩn đoán và xử lý ảnh..."):
                    res_auto, steps_taken, reasons = auto_enhance_image(curr_img)
                    apply_step(st.session_state, res_auto, "Tự động cải thiện (Auto Enhance)", {"steps": steps_taken})

                    st.success("Đã hoàn tất tự động cải thiện ảnh!")
                    st.markdown("###### CÁC BƯỚC ĐÃ THỰC HIỆN:")
                    for s in steps_taken:
                        st.markdown(f"- **{s}**")
                    st.markdown("###### LÝ DO CHUYÊN MÔN:")
                    for r in reasons:
                        st.markdown(f"- 💡 *{r}*")
                    st.rerun()

        # TAB 7: LỊCH SỬ CHỈNH SỬA & SO SÁNH TỪNG BƯỚC (MỤC XXI & XXVIII)
        with tab_hist_log:
            st.markdown("##### Lịch sử các bước xử lý đã thực hiện")
            hist_list = st.session_state.get("history", [])

            if hist_list:
                step_names = [f"Bước {h['step']}: {h['name']} ({h['time']})" for h in hist_list]
                selected_step_idx = st.selectbox("Chọn bước trong lịch sử để xem lại ảnh:", range(len(hist_list)), format_func=lambda i: step_names[i])

                viewed_hist_item = hist_list[selected_step_idx]
                st.image(viewed_hist_item["image"], caption=f"Ảnh tại {step_names[selected_step_idx]}", use_container_width=True)

                if st.button("Khôi phục ảnh về thời điểm bước này", help="Đặt ảnh tại bước này làm ảnh hiện tại"):
                    st.session_state["current_image"] = viewed_hist_item["image"].copy()
                    st.session_state["preview_image"] = None
                    st.success(f"Đã khôi phục ảnh về: {viewed_hist_item['name']}")
                    st.rerun()


# =============================================================================
# 3. HISTOGRAM & PHÂN TÍCH
# =============================================================================
elif menu_choice == "3. Histogram & phân tích":
    st.header("📈 Phân Tích Histogram Của Ảnh Hiện Tại")
    if require_image():
        curr_img = st.session_state["current_image"]
        orig_img = st.session_state["original_image"]

        st.markdown(
            "> 📌 **Nguyên tắc:** Biểu đồ Histogram dưới đây được tính toán trực tiếp trên **`current_image`** "
            "(đã kế thừa toàn bộ các bước chỉnh sửa hiện tại) và cho phép đối chiếu với ảnh gốc ban đầu."
        )

        col_img_h, col_chart_h = st.columns([1, 1.2])
        with col_img_h:
            st.image(curr_img, caption=f"Ảnh hiện tại ({st.session_state['history'][-1]['name']})", use_container_width=True)

        with col_chart_h:
            is_col = (curr_img.ndim == 3 and curr_img.shape[2] == 3)
            h_choice = st.radio(
                "Lựa chọn kênh Histogram:",
                ["RGB (Đỏ, Lục, Lam)", "Grayscale (Mức xám 0-255)", "Kênh Đỏ (Red)", "Kênh Lục (Green)", "Kênh Lam (Blue)"] if is_col else ["Grayscale (0-255)"],
                horizontal=True,
            )

            # Vẽ theo lựa chọn
            fig_h, ax_h = plt.subplots(figsize=(7, 3.8), dpi=100)
            if "RGB" in h_choice and is_col:
                colors = ("red", "green", "blue")
                labels = ("Kênh Red", "Kênh Green", "Kênh Blue")
                for i, (c, l) in enumerate(zip(colors, labels)):
                    hist_c = calculate_histogram(curr_img, is_gray=False)
                    ax_h.plot(hist_c[colors[i]], color=c, alpha=0.85, linewidth=1.5, label=l)
                    ax_h.fill_between(range(256), hist_c[colors[i]], color=c, alpha=0.15)
            elif "Đỏ" in h_choice and is_col:
                h_c = calculate_histogram(curr_img, is_gray=False)["red"]
                ax_h.plot(h_c, color="red", linewidth=1.8, label="Kênh Red")
                ax_h.fill_between(range(256), h_c, color="red", alpha=0.2)
            elif "Lục" in h_choice and is_col:
                h_c = calculate_histogram(curr_img, is_gray=False)["green"]
                ax_h.plot(h_c, color="green", linewidth=1.8, label="Kênh Green")
                ax_h.fill_between(range(256), h_c, color="green", alpha=0.2)
            elif "Lam" in h_choice and is_col:
                h_c = calculate_histogram(curr_img, is_gray=False)["blue"]
                ax_h.plot(h_c, color="blue", linewidth=1.8, label="Kênh Blue")
                ax_h.fill_between(range(256), h_c, color="blue", alpha=0.2)
            else:
                h_gray = calculate_histogram(curr_img, is_gray=True)["gray"]
                ax_h.plot(h_gray, color="#2563EB", linewidth=1.8, label="Mức xám (Grayscale)")
                ax_h.fill_between(range(256), h_gray, color="#93C5FD", alpha=0.4)

            ax_h.set_title("Lược Đồ Histogram (Ảnh Hiện Tại)", fontweight="bold")
            ax_h.set_xlabel("Mức xám / Cường độ pixel (0 - 255)")
            ax_h.set_ylabel("Số lượng điểm ảnh")
            ax_h.set_xlim([0, 256])
            ax_h.grid(True, linestyle=":", alpha=0.6)
            ax_h.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

        st.markdown("---")
        st.subheader("🔢 Thông Số Thống Kê Định Lượng")
        stats_curr = calculate_statistics(curr_img)
        stats_orig = calculate_statistics(orig_img)
        entropy_curr = calculate_entropy(curr_img)
        entropy_orig = calculate_entropy(orig_img)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Min Pixel", f"{stats_curr['min']:.0f}", delta=f"{stats_curr['min'] - stats_orig['min']:+.0f}")
        m2.metric("Max Pixel", f"{stats_curr['max']:.0f}", delta=f"{stats_curr['max'] - stats_orig['max']:+.0f}")
        m3.metric("Độ sáng TB (Mean)", f"{stats_curr['mean']}", delta=f"{stats_curr['mean'] - stats_orig['mean']:+.2f}")
        m4.metric("Độ lệch chuẩn (Std)", f"{stats_curr['std']}", delta=f"{stats_curr['std'] - stats_orig['std']:+.2f}")
        m5.metric("Trung vị (Median)", f"{stats_curr['median']}", delta=f"{stats_curr['median'] - stats_orig['median']:+.1f}")
        m6.metric("Entropy", f"{entropy_curr:.3f}", delta=f"{entropy_curr - entropy_orig:+.3f}")

        st.markdown("---")
        st.subheader("🤖 Tự Động Phân Tích & Chẩn Đoán Ảnh Hiện Tại")
        diag = analyze_histogram(curr_img)
        for d in diag["details"]:
            st.markdown(f"- 📌 {d}")
        if diag["recommendations"]:
            st.markdown("**Gợi ý kỹ thuật tiếp theo:**")
            for r in diag["recommendations"]:
                st.markdown(f"- 💡 {r}")


# =============================================================================
# 4. SO SÁNH THUẬT TOÁN (CHẠY ĐỘC LẬP TỪ ORIGINAL_IMAGE)
# =============================================================================
elif menu_choice == "4. So sánh thuật toán":
    st.header("⚖️ So Sánh Các Thuật Toán (Khảo Sát Độc Lập Từ Ảnh Gốc)")
    if require_image():
        orig_img = st.session_state["original_image"]
        st.markdown(
            """
            > 🎯 **Nguyên tắc so sánh công bằng:** Toàn bộ các thuật toán dưới đây đều được chạy **ĐỘC LẬP từ cùng một ảnh gốc (`original_image`)**, 
            > không bị phụ thuộc vào ảnh đang chỉnh sửa (`current_image`). Đồng thời hệ thống đo lường chính xác thời gian thực thi (Performance Timing) của từng thuật toán.
            """
        )

        algo_options = [
            "Histogram Equalization",
            "CLAHE (Clip=2.0, Grid=8x8)",
            "Brightness (+30)",
            "Contrast (1.4x)",
            "Gamma Correction (0.75)",
            "Khử nhiễu Bilateral Filter",
            "Làm nét Unsharp Mask",
        ]

        selected_algos = st.multiselect(
            "Chọn 2 đến 5 thuật toán muốn đối chiếu:",
            options=algo_options,
            default=["Histogram Equalization", "CLAHE (Clip=2.0, Grid=8x8)", "Brightness (+30)", "Làm nét Unsharp Mask"],
        )

        if len(selected_algos) < 2:
            st.warning("Vui lòng chọn ít nhất 2 thuật toán để tiến hành đối chiếu.")
        else:
            timing_data = {}
            results_dict = {"Ảnh gốc (Original)": orig_img}
            table_records = []

            for algo in selected_algos:
                t_start = time.perf_counter()
                if algo == "Histogram Equalization":
                    res = equalize_histogram_opencv(orig_img)
                elif "CLAHE" in algo:
                    res = apply_clahe(orig_img, clip_limit=2.0, tile_grid_size=(8, 8))
                elif "Brightness" in algo:
                    res = adjust_brightness(orig_img, beta=30)
                elif "Contrast" in algo:
                    res = adjust_contrast(orig_img, alpha=1.4)
                elif "Gamma" in algo:
                    res = adjust_gamma(orig_img, gamma=0.75)
                elif "Bilateral" in algo:
                    res = apply_denoising(orig_img, method="Bilateral Filter", level="Vừa")
                else:  # Sharpen
                    res = sharpen_image(orig_img, strength=45, method="Unsharp Mask")

                t_cost = (time.perf_counter() - t_start) * 1000.0  # ms
                timing_data[algo] = t_cost
                results_dict[algo] = res

                # Tính chỉ số
                ev = evaluate_all(orig_img, res)
                table_records.append({
                    "Thuật toán": algo,
                    "Thời gian xử lý": f"{t_cost:.2f} ms",
                    "Mean": ev["enhanced"]["mean"],
                    "Std (Tương phản)": ev["enhanced"]["std"],
                    "Entropy (Chi tiết)": ev["enhanced"]["entropy"],
                    "PSNR": ev["comparison"]["psnr"],
                    "SSIM": ev["comparison"]["ssim"],
                })

            st.markdown("---")
            st.subheader("🖼️ Trực Quan Hóa Kết Quả Các Thuật Toán")
            cols_grid = st.columns(min(4, len(results_dict)))
            for idx, (name, img_out) in enumerate(results_dict.items()):
                c_target = cols_grid[idx % len(cols_grid)]
                with c_target:
                    st.markdown(f"**{name}**")
                    st.image(img_out, use_container_width=True)

            st.markdown("---")
            st.subheader("⏱️ Bảng Hiệu Năng & Chỉ Số So Sánh Công Bằng")
            st.dataframe(pd.DataFrame(table_records), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Lược Đồ Histogram Đa Thuật Toán")
            fig_multi_c = plot_multi_histogram(results_dict)
            if fig_multi_c:
                st.pyplot(fig_multi_c)
                plt.close(fig_multi_c)


# =============================================================================
# 5. ĐÁNH GIÁ KẾT QUẢ (ORIGINAL VS CURRENT)
# =============================================================================
elif menu_choice == "5. Đánh giá kết quả":
    st.header("📊 Đánh Giá Định Lượng Chất Lượng Ảnh (Original VS Current)")
    if require_image():
        orig_img = st.session_state["original_image"]
        curr_img = st.session_state["current_image"]

        ev_final = evaluate_all(orig_img, curr_img)

        st.subheader("📋 Bảng So Sánh Chỉ Số Định Lượng")
        m_rows = [
            {
                "Chỉ số đánh giá": "Mean (Độ sáng trung bình)",
                "Ảnh Gốc (Original)": ev_final["original"]["mean"],
                "Ảnh Hiện Tại (Current)": ev_final["enhanced"]["mean"],
                "Độ chênh lệch": f"{ev_final['comparison']['mean_diff']:+.2f}",
                "Ý nghĩa": "Phản ánh mức độ sáng tổng quan của ảnh trong dải [0, 255]",
            },
            {
                "Chỉ số đánh giá": "Standard Deviation - Std (Tương phản)",
                "Ảnh Gốc (Original)": ev_final["original"]["std"],
                "Ảnh Hiện Tại (Current)": ev_final["enhanced"]["std"],
                "Độ chênh lệch": f"{ev_final['comparison']['std_diff']:+.2f}",
                "Ý nghĩa": "Độ phân tán mức xám; Std càng cao thì tương phản càng rõ rệt",
            },
            {
                "Chỉ số đánh giá": "Entropy Shannon (Độ phong phú thông tin)",
                "Ảnh Gốc (Original)": ev_final["original"]["entropy"],
                "Ảnh Hiện Tại (Current)": ev_final["enhanced"]["entropy"],
                "Độ chênh lệch": f"{ev_final['comparison']['entropy_diff']:+.3f}",
                "Ý nghĩa": "Đo lượng thông tin và độ chi tiết ẩn chứa trong các mức xám (bits)",
            },
            {
                "Chỉ số đánh giá": "PSNR (Peak Signal-to-Noise Ratio)",
                "Ảnh Gốc (Original)": "Chuẩn (Tham chiếu)",
                "Ảnh Hiện Tại (Current)": ev_final["comparison"]["psnr"],
                "Độ chênh lệch": "-",
                "Ý nghĩa": "Tỷ số tín hiệu cực đại trên sai số bình phương trung bình (dB)",
            },
            {
                "Chỉ số đánh giá": "SSIM (Structural Similarity Index)",
                "Ảnh Gốc (Original)": "1.0000",
                "Ảnh Hiện Tại (Current)": f"{ev_final['comparison']['ssim']:.4f}",
                "Độ chênh lệch": f"{ev_final['comparison']['ssim'] - 1.0:+.4f}",
                "Ý nghĩa": "Mức độ bảo toàn cấu trúc hình học và đặc trưng nhận dạng của vật thể",
            },
        ]
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Biểu Đồ Cột Đối Chiếu Matplotlib")
        chart_input = {
            "Ảnh Gốc (Original)": {
                "mean": ev_final["original"]["mean"],
                "std": ev_final["original"]["std"],
                "entropy": ev_final["original"]["entropy"]
            },
            "Ảnh Đã Nâng Cấp (Current)": {
                "mean": ev_final["enhanced"]["mean"],
                "std": ev_final["enhanced"]["std"],
                "entropy": ev_final["enhanced"]["entropy"]
            }
        }
        fig_ev_bar = plot_evaluation_metrics_bar(chart_input)
        st.pyplot(fig_ev_bar)
        plt.close(fig_ev_bar)

        st.markdown("---")
        st.subheader("📝 Nhận Xét Chuyên Môn Về Sự Thay Đổi Chất Lượng:")
        for n in ev_final["notes"]:
            st.markdown(f"- ✅ **{n}**")


# =============================================================================
# 6. SO SÁNH BEFORE / AFTER
# =============================================================================
elif menu_choice == "6. So sánh Before / After":
    st.header("🪞 So Sánh Trực Quan Before / After")
    if require_image():
        orig_img = st.session_state["original_image"]
        curr_img = st.session_state["current_image"]

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("### ◀ BEFORE (ẢNH GỐC)")
            st.image(orig_img, caption="Ảnh gốc ban đầu", use_container_width=True)

        with col_a:
            last_op_name = st.session_state["history"][-1]["name"] if st.session_state.get("history") else "Current"
            st.markdown(f"### ▶ AFTER (ẢNH ĐÃ XỬ LÝ) — `{last_op_name}`")
            st.image(curr_img, caption="Ảnh hoàn thiện sau toàn bộ chuỗi chỉnh sửa", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Histogram Before VS Histogram After")
        fig_ba = plot_comparison_histograms(orig_img, curr_img, title1="Histogram Before (Ảnh Gốc)", title2="Histogram After (Ảnh Hiện Tại)")
        st.pyplot(fig_ba)
        plt.close(fig_ba)


# =============================================================================
# 7. DEMO TỰ ĐỘNG & THỰC NGHIỆM
# =============================================================================
elif menu_choice == "7. Demo tự động & Thực nghiệm":
    st.header("🚀 Demo Tự Động Toàn Quy Trình & Thực Nghiệm Báo Cáo")
    if require_image():
        st.markdown(
            """
            Chế độ này phục vụ **thuyết trình trực tiếp trên lớp**:
            Hệ thống sẽ thực hiện tuần tự chuỗi xử lý liên tiếp:
            `Ảnh gốc` ➜ `Brightness` ➜ `Contrast` ➜ `CLAHE` ➜ `Denoising` ➜ `Sharpen` ➜ `Kết quả hoàn thiện`
            """
        )

        if st.button("▶ BẮT ĐẦU CHẠY DEMO TỰ ĐỘNG", type="primary"):
            steps_container = st.container()
            with steps_container:
                st.info("Bắt đầu thực thi chuỗi nâng cấp liên tiếp...")

                # Bước 1
                st.markdown("#### Bước 1: Ảnh Gốc & Khảo Sát Histogram")
                step_img = st.session_state["original_image"].copy()
                st.image(step_img, width=450)

                # Bước 2: Brightness
                st.markdown("#### Bước 2: Tăng Độ Sáng (Brightness β = +25)")
                step_img = adjust_brightness(step_img, beta=25)
                st.image(step_img, width=450)

                # Bước 3: Contrast
                st.markdown("#### Bước 3: Tăng Độ Tương Phản (Contrast α = 1.25)")
                step_img = adjust_contrast(step_img, alpha=1.25)
                st.image(step_img, width=450)

                # Bước 4: CLAHE
                st.markdown("#### Bước 4: Tăng Cường Vi Tương Phản Cục Bộ (CLAHE Clip=2.0)")
                step_img = apply_clahe(step_img, clip_limit=2.0, tile_grid_size=(8, 8))
                st.image(step_img, width=450)

                # Bước 5: Khử nhiễu
                st.markdown("#### Bước 5: Khử Nhiễu Giữ Cạnh (Bilateral Filter)")
                step_img = apply_denoising(step_img, method="Bilateral Filter", level="Nhẹ")
                st.image(step_img, width=450)

                # Bước 6: Làm nét
                st.markdown("#### Bước 6: Làm Nét Viền Cạnh (Unsharp Mask +35)")
                step_img = sharpen_image(step_img, strength=35, method="Unsharp Mask")
                st.image(step_img, width=450)

                st.success("✅ Hoàn thành toàn bộ chuỗi Demo tự động!")
                if st.button("Lưu kết quả Demo này thành `current_image`"):
                    apply_step(st.session_state, step_img, "Demo tự động (Chuỗi liên hoàn)")
                    st.rerun()


# =============================================================================
# 8. CƠ SỞ LÝ THUYẾT (MỤC XXXII)
# =============================================================================
elif menu_choice == "8. Cơ sở lý thuyết":
    st.header("📖 Cơ Sở Lý Thuyết & Nguyên Lý Khoa Học Toàn Diện")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        [
            "1. Ảnh số & Histogram",
            "2. Histogram Equalization & CDF",
            "3. CLAHE",
            "4. Brightness & Contrast",
            "5. Khử nhiễu (Denoising)",
            "6. Làm nét & Deblur",
            "7. Màu sắc & Cân bằng trắng",
        ]
    )

    with t1:
        st.markdown(
            """
            ### 1. Khái niệm Ảnh số và Lược đồ mức xám (Histogram)
            - **Ảnh số (Digital Image):** Là một ma trận hai chiều $f(x, y)$ kích thước $M \\times N$, trong đó giá trị tại mỗi tọa độ $(x, y)$ là một điểm ảnh (**Pixel**) biểu diễn cường độ sáng. Với ảnh 8-bit, mức sáng rời rạc nằm trong khoảng $[0, L - 1]$ với $L = 256$.
            - **Lược đồ mức xám (Histogram):** Là hàm rời rạc $h(r_k) = n_k$, với $n_k$ là số pixel có mức xám $r_k$.
            - **Hàm mật độ xác suất (PDF):** $p(r_k) = \\frac{n_k}{M \\times N}$ thỏa mãn $\\sum_{k=0}^{L-1} p(r_k) = 1$.
            """
        )

    with t2:
        st.markdown(
            """
            ### 2. Cân bằng Histogram (Histogram Equalization - HE)
            - **Mục đích:** Tìm hàm biến đổi $s = T(r)$ sao cho phân bố mức xám đầu ra là phân bố đều (Uniform Distribution) trên $[0, 255]$, từ đó tối đa hóa độ tương phản toàn cục.
            - **Hàm tích lũy (CDF):**
              $$s_k = T(r_k) = (L - 1) \\sum_{j=0}^{k} p(r_j) = \\text{round}\\left( \\frac{CDF(r_k) - CDF_{min}}{(M \\times N) - CDF_{min}} \\times 255 \\right)$$
            - **Xử lý ảnh màu:** Phải chuyển sang không gian **YCrCb** và chỉ cân bằng trên kênh độ sáng $Y$. Cân bằng trực tiếp trên $R, G, B$ sẽ làm lệch tỉ lệ màu sắc tự nhiên, gây ra lỗi méo màu (Color Distortion).
            """
        )

    with t3:
        st.markdown(
            """
            ### 3. Thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization)
            - **Vấn đề của HE:** Khi ảnh có các vùng đồng nhất lớn (bầu trời, nền phẳng), histogram tại đó có đỉnh rất nhọn, HE sẽ khuếch đại nhiễu cực kỳ nghiêm trọng.
            - **Nguyên lý CLAHE:**
              1. Chia ảnh thành các ô cục bộ (**Tiles**), thông thường kích thước $8 \\times 8$.
              2. Cắt đỉnh histogram vượt ngưỡng (**Clip Limit**); số pixel vượt ngưỡng được phân phối đều cho các cột khác.
              3. Cân bằng histogram độc lập trong từng ô.
              4. **Nội suy song tuyến tính (Bilinear Interpolation):** Khử ranh giới viền ô giữa các khối lân cận.
            """
        )

    with t4:
        st.markdown(
            """
            ### 4. Điều chỉnh Tuyến tính: Brightness & Contrast
            - **Brightness (Độ sáng):** $I' = \\text{clip}(I + \\beta, 0, 255)$. Tịnh tiến phân bố histogram sang trái/phải.
            - **Contrast (Tương phản):** $I' = \\text{clip}(\\alpha \\times I, 0, 255)$. Kéo dãn ($\\alpha > 1$) hoặc nén ($\\alpha < 1$) dải mức xám.
            - **Gamma Correction:** $I' = 255 \\times (I / 255)^\\gamma$. Kéo sáng vùng tối phi tuyến mà không làm bão hòa vùng sáng.
            """
        )

    with t5:
        st.markdown(
            """
            ### 5. Khử nhiễu (Image Denoising)
            - **Gaussian Blur:** Tích chập với nhân phân bố chuẩn, làm mượt ảnh nhưng làm mờ cạnh.
            - **Median Filter:** Thay thế pixel bằng giá trị trung vị của lân cận, loại bỏ hoàn toàn nhiễu muối tiêu (Salt & Pepper Noise).
            - **Bilateral Filter:** Lọc song phương kết hợp cả khoảng cách không gian lẫn chênh lệch cường độ sáng:
              $$I_{filtered}(x) = \\frac{1}{W_p} \\sum_{x_i \\in \\Omega} I(x_i) f_r(\\|I(x_i) - I(x)\\|) g_s(\\|x_i - x\\|)$$
              Giữ nguyên độ sắc nét của đường biên cạnh!
            - **Non-Local Means:** So sánh độ tương đồng giữa các mảng điểm ảnh (patches) trên toàn vùng tìm kiếm để triệt tiêu nhiễu với độ chính xác cao nhất.
            """
        )

    with t6:
        st.markdown(
            """
            ### 6. Làm nét (Sharpening) & Phục hồi ảnh mờ (Deblurring)
            - **Unsharp Masking:** $I_{sharp} = I + k \\times (I - I_{blur})$.
            - **Toán tử Laplace:** $I_{sharp} = I - \\nabla^2 I$.
            - **Deblur:**
              - Giải chập Richardson-Lucy với hàm truyền điểm PSF (Point Spread Function) Gaussian.
              - Bộ lọc Wiener cân bằng giữa việc phục hồi tín hiệu và khống chế nhiễu tần số cao.
            """
        )

    with t7:
        st.markdown(
            """
            ### 7. Màu sắc & Cân bằng trắng (White Balance)
            - **Không gian màu HSV:** Tách biệt Sắc màu ($H$), Độ bão hòa ($S$), và Cường độ sáng ($V$).
            - **Giả thuyết Gray World:** Trung bình cường độ các kênh màu trong ảnh tự nhiên có xu hướng bằng nhau:
              $$\\mu_{gray} = \\frac{\\mu_R + \\mu_G + \\mu_B}{3}, \\quad R' = R \\times \\frac{\\mu_{gray}}{\\mu_R}$$
            - Khắc phục triệt để hiện tượng ám vàng (đèn sợi đốt) hoặc ám xanh (bóng râm).
            """
        )
