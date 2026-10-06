"""
Module: utils/visualization.py
Mục đích:
1. Vẽ biểu đồ Histogram (Grayscale, RGB) chuyên nghiệp với Matplotlib.
2. Vẽ hàm tích lũy CDF (Cumulative Distribution Function).
3. Vẽ so sánh Histogram trước và sau xử lý trực quan.
4. Vẽ biểu đồ cột so sánh các chỉ số định lượng (Mean, Std, Entropy) giữa các thuật toán.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def _configure_matplotlib_style():
    """Cấu hình phông chữ và giao diện matplotlib cho sáng sủa, sắc nét"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def plot_histogram_matplotlib(img_np, hist_type="auto", title="Biểu đồ Histogram", show_cdf=False):
    """
    Vẽ biểu đồ Histogram của ảnh.
    
    Args:
        img_np (np.ndarray): Mảng ảnh đầu vào.
        hist_type (str): 'auto', 'gray', hoặc 'rgb'.
        title (str): Tiêu đề biểu đồ.
        show_cdf (bool): Có hiển thị đường cong phân bố tích lũy (CDF) hay không.
        
    Returns:
        matplotlib.figure.Figure: Đối tượng figure để hiển thị bằng st.pyplot(fig).
    """
    _configure_matplotlib_style()

    is_color = (img_np.ndim == 3 and img_np.shape[2] == 3)
    if hist_type == "auto":
        hist_type = "rgb" if is_color else "gray"

    fig, ax1 = plt.subplots(figsize=(7, 3.8), dpi=100)

    if hist_type == "rgb" and is_color:
        # Vẽ 3 kênh màu Red, Green, Blue
        colors = ("red", "green", "blue")
        channel_names = ("Kênh Đỏ (R)", "Kênh Lục (G)", "Kênh Lam (B)")
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            hist = cv2.calcHist([img_np], [i], None, [256], [0, 256]).flatten()
            ax1.plot(hist, color=color, alpha=0.85, linewidth=1.5, label=name)
            ax1.fill_between(range(256), hist, color=color, alpha=0.15)
    else:
        # Vẽ kênh mức xám
        if is_color:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            gray = img_np[:, :, 0]
        else:
            gray = img_np

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        ax1.plot(hist, color="#2b5c8f", linewidth=1.8, label="Mức xám (Grayscale)")
        ax1.fill_between(range(256), hist, color="#4a90e2", alpha=0.35)

        if show_cdf:
            ax2 = ax1.twinx()
            cdf = hist.cumsum()
            cdf_norm = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            ax2.plot(cdf_norm, color="#d9534f", linestyle="--", linewidth=2.0, label="Hàm CDF")
            ax2.set_ylabel("Xác suất tích lũy CDF", color="#d9534f")
            ax2.tick_params(axis="y", labelcolor="#d9534f")
            ax2.set_ylim([0, 1.05])
            # Gộp legend cả 2 trục
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    ax1.set_title(title, fontweight="bold", pad=10)
    ax1.set_xlabel("Mức xám / Cường độ pixel (0 - 255)")
    ax1.set_ylabel("Số lượng điểm ảnh (Pixel count)")
    ax1.set_xlim([0, 256])
    ax1.grid(True, linestyle=":", alpha=0.6)
    if not (show_cdf and not (hist_type == "rgb" and is_color)):
        ax1.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_comparison_histograms(img1, img2, title1="Histogram Ảnh Gốc", title2="Histogram Ảnh Sau Xử Lý", hist_type="gray"):
    """
    Vẽ 2 biểu đồ Histogram cạnh nhau để so sánh trực quan sự phân bố mức xám.
    """
    _configure_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=100)

    for ax, img, title in zip((ax1, ax2), (img1, img2), (title1, title2)):
        is_color = (img.ndim == 3 and img.shape[2] == 3)
        if hist_type == "rgb" and is_color:
            colors = ("red", "green", "blue")
            channel_names = ("Đỏ", "Lục", "Lam")
            for i, (color, name) in enumerate(zip(colors, channel_names)):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
                ax.plot(hist, color=color, alpha=0.85, linewidth=1.4, label=name)
                ax.fill_between(range(256), hist, color=color, alpha=0.12)
        else:
            if is_color:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[2] == 1:
                gray = img[:, :, 0]
            else:
                gray = img

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            ax.plot(hist, color="#1e3d59", linewidth=1.6)
            ax.fill_between(range(256), hist, color="#17b978", alpha=0.35)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cường độ pixel (0 - 255)")
        ax.set_ylabel("Số pixel")
        ax.set_xlim([0, 256])
        ax.grid(True, linestyle=":", alpha=0.6)
        if hist_type == "rgb" and is_color:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_evaluation_metrics_bar(methods_data):
    """
    Vẽ biểu đồ cột so sánh các chỉ số định lượng: Mean, Std, Entropy giữa các phương pháp.
    
    Args:
        methods_data (dict): Dict dạng {
            'Tên phương pháp': {'mean': 120.5, 'std': 45.2, 'entropy': 6.8}, ...
        }
    """
    _configure_matplotlib_style()

    labels = list(methods_data.keys())
    means = [data["mean"] for data in methods_data.values()]
    stds = [data["std"] for data in methods_data.values()]
    entropies = [data["entropy"] for data in methods_data.values()]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax1 = plt.subplots(figsize=(10, 4.2), dpi=100)

    # Cột Mean và Std theo thang trục trái [0 - 255]
    rects1 = ax1.bar(x - width, means, width, label="Độ sáng TB (Mean)", color="#4361ee", alpha=0.85)
    rects2 = ax1.bar(x, stds, width, label="Độ lệch chuẩn (Std)", color="#4cc9f0", alpha=0.85)

    ax1.set_ylabel("Thang đo cường độ sáng [0 - 255]")
    ax1.set_title("So sánh định lượng giữa các phương pháp xử lý", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontweight="semibold")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Entropy theo thang trục phải [0 - 8 bits]
    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, entropies, width, label="Entropy (bits)", color="#f72585", alpha=0.85)
    ax2.set_ylabel("Entropy Shannon [0 - 8 bits]", color="#f72585")
    ax2.tick_params(axis="y", labelcolor="#f72585")
    ax2.set_ylim([0, 8.5])

    # Gộp legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=3)

    plt.tight_layout()
    return fig


def plot_multi_histogram(images_dict, hist_type="gray"):
    """
    Vẽ nhiều biểu đồ Histogram trên cùng một lưới để so sánh đa phương pháp.
    Args:
        images_dict (dict): {'Tên phương pháp': img_np}
    """
    _configure_matplotlib_style()
    n = len(images_dict)
    if n == 0:
        return None

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), dpi=100)
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, (name, img) in enumerate(images_dict.items()):
        ax = axes_flat[idx]
        is_color = (img.ndim == 3 and img.shape[2] == 3)
        if hist_type == "rgb" and is_color:
            colors = ("red", "green", "blue")
            for c_i, color in enumerate(colors):
                h = cv2.calcHist([img], [c_i], None, [256], [0, 256]).flatten()
                ax.plot(h, color=color, alpha=0.8, linewidth=1.2)
        else:
            if is_color:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[2] == 1:
                gray = img[:, :, 0]
            else:
                gray = img
            h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            ax.plot(h, color="#3a0ca3", linewidth=1.5)
            ax.fill_between(range(256), h, color="#7209b7", alpha=0.25)

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlim([0, 256])
        ax.grid(True, linestyle=":", alpha=0.5)

    # Ẩn các ô trống nếu có
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    return fig
