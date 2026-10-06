# HỆ THỐNG PHÂN TÍCH VÀ TĂNG CƯỜNG CHẤT LƯỢNG ẢNH BẰNG HISTOGRAM

> **Đề tài:** Histogram và tăng cường chất lượng ảnh  
> **Ngôn ngữ:** Python 3.x  
> **Giao diện:** Streamlit Web Application  
> **Môi trường phát triển:** Visual Studio Code (Windows)

---

## 1. TÊN ĐỀ TÀI & GIỚI THIỆU

**"HISTOGRAM VÀ TĂNG CƯỜNG CHẤT LƯỢNG ẢNH"**

Ứng dụng là một hệ thống phần mềm hoàn chỉnh phục vụ nghiên cứu và thực hành xử lý ảnh số. Khác với các công cụ minh họa độc lập, hệ thống được xây dựng theo **mô hình xử lý ảnh liên tục tuần hoàn**:
- **`original_image` (Ảnh gốc):** Được bảo toàn nguyên vẹn trong suốt phiên làm việc.
- **`current_image` (Ảnh hiện tại):** Là thực thể chỉnh sửa liên tục, mọi kỹ thuật áp dụng đều kế thừa từ kết quả của các bước trước đó.
- **Ngăn xếp Undo / Redo & Lịch sử chỉnh sửa:** Cho phép quay lại, làm lại hoặc khôi phục trạng thái bất kỳ lúc nào.

```
original_image
      ↓
current_image ➜ Brightness ➜ current_image ➜ Contrast ➜ current_image ➜ CLAHE ➜ Denoising ➜ Sharpen ➜ Hoàn thiện
```

---

## 2. MỤC TIÊU DỰ ÁN

1. Cho phép người dùng tải lên ảnh đa định dạng (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. Phân tích chi tiết và vẽ biểu đồ Histogram (Grayscale, RGB, từng kênh R, G, B, CDF).
3. Cho phép chỉnh sửa liên tiếp trên cùng một ảnh:
   - Điều chỉnh độ sáng (Brightness)
   - Điều chỉnh độ tương phản (Contrast)
   - Cân bằng lược đồ mức xám (Histogram Equalization)
   - Cân bằng tương phản thích nghi cục bộ (CLAHE)
   - Hiệu chỉnh Gamma, Exposure, Highlights & Shadows
   - Khử nhiễu đa cấp độ (Gaussian, Median, Bilateral, Non-Local Means)
   - Làm nét ảnh (Sharpening - Unsharp Mask, High-pass, Kernel)
   - Cải thiện ảnh mờ (Deblur - Wiener, Richardson-Lucy)
   - Tinh chỉnh màu sắc (Saturation, Hue, Temperature, White Balance Gray World)
   - Tự động cải thiện ảnh (Auto Enhance)
4. Hỗ trợ xem trước (Preview) và áp dụng (Apply) trước khi ghi nhận vào ảnh hiện tại.
5. Hỗ trợ Hoàn tác (Undo), Làm lại (Redo), Khôi phục ảnh gốc (Reset) và xem lịch sử chỉnh sửa từng bước.
6. So sánh Before / After và đối chiếu độc lập đa thuật toán từ ảnh gốc cùng bảng đo hiệu năng thời gian xử lý.
7. Đánh giá chất lượng định lượng: Mean, Std, Entropy, PSNR, SSIM.
8. Xuất và tải ảnh kết quả chất lượng cao định dạng PNG hoặc JPG.

---

## 3. NỘI DUNG NGHIÊN CỨU

Hệ thống triển khai 8 nhóm công nghệ cốt lõi:
1. **Phân tích Histogram:** Khảo sát phân bố mật độ mức xám, chẩn đoán tự động ảnh thiếu sáng, dư sáng, tương phản kém.
2. **Ánh sáng & Tương phản:** Biến đổi tuyến tính $I' = \text{clip}(\alpha I + \beta, 0, 255)$ và phi tuyến Gamma Correction $I' = 255 \times (I / 255)^\gamma$.
3. **Cân bằng Histogram:** Cân bằng toàn cục HE và cân bằng cục bộ CLAHE bảo toàn màu sắc qua không gian YCrCb và LAB.
4. **Lọc không gian & Khử nhiễu:** Gaussian Blur, lọc trung vị Median Filter, lọc song phương Bilateral Filter, Non-Local Means Denoising.
5. **Làm nét & Chi tiết:** Unsharp Masking, High-pass, Sobel Edge Enhancement, Detail Enhancement.
6. **Phục hồi ảnh mờ:** Giải chập Richardson-Lucy PSF Gaussian và lọc nghịch đảo Wiener.
7. **Xử lý màu sắc:** Không gian màu HSV (Saturation, Hue), cân bằng trắng Gray World / White Patch.
8. **Đánh giá chất lượng ảnh:** Các chỉ số khách quan thống kê và cấu trúc (Mean, Std, Entropy, PSNR, SSIM).

---

## 4. CÔNG NGHỆ SỬ DỤNG

* **Python 3.10+ / 3.13:** Ngôn ngữ phát triển chính.
* **OpenCV (`opencv-python`):** Xử lý ma trận ảnh, chuyển đổi không gian màu, lọc không gian, CLAHE, NL-Means.
* **NumPy:** Thao tác đại số tuyến tính, tính toán mảng và tự cài đặt thuật toán CDF giải tích.
* **Matplotlib:** Vẽ biểu đồ Histogram độ phân giải cao và biểu đồ cột đối chiếu đa chỉ số.
* **Pillow (`PIL`):** Xử lý xuất nhập định dạng ảnh an toàn.
* **scikit-image (`skimage`):** Tính toán PSNR, SSIM, giải chập Richardson-Lucy và Wiener.
* **Pandas:** Bảng biểu dữ liệu thống kê và hiệu năng thuật toán.
* **Streamlit:** Framework giao diện web trực quan, tương tác thời gian thực.

---

## 5. CẤU TRÚC PROJECT

```text
histogram-image-enhancement/
│
├── app.py                      # Ứng dụng web Streamlit & điều phối quy trình liên tục
├── requirements.txt            # Danh sách thư viện cần thiết
├── README.md                   # Tài liệu báo cáo dự án
├── run.bat                     # File kích hoạt 1-click trên Windows
│
├── modules/                    # Các module thuật toán chuyên biệt
│   ├── __init__.py
│   ├── histogram.py            # Tính toán histogram, CDF, thống kê và tự động chẩn đoán
│   ├── equalization.py         # Histogram Equalization (OpenCV chuẩn YCrCb & NumPy CDF)
│   ├── clahe.py                # CLAHE với tùy biến Clip Limit & Tile Grid Size
│   ├── brightness.py           # Điều chỉnh độ sáng I_new = I + β
│   ├── contrast.py             # Điều chỉnh độ tương phản I_new = α × I
│   ├── gamma.py                # Gamma Correction, Exposure, Highlights & Shadows
│   ├── sharpening.py           # Làm nét (Unsharp Mask, Kernel, High-pass, Edge/Detail)
│   ├── denoising.py            # Khử nhiễu (Gaussian, Median, Bilateral, NL-Means)
│   ├── deblurring.py           # Phục hồi ảnh mờ (Richardson-Lucy, Wiener, Unsharp)
│   ├── color.py                # Saturation, Hue, Temperature, Sepia, Grayscale
│   ├── white_balance.py        # Cân bằng trắng Gray World & White Patch
│   ├── auto_enhance.py         # Thuật toán tự động chẩn đoán và tăng cường tối ưu
│   └── evaluation.py           # Đánh giá định lượng Mean, Std, Entropy, PSNR, SSIM
│
├── utils/                      # Tiện ích bổ trợ
│   ├── __init__.py
│   ├── image_utils.py          # Đọc ảnh, trích xuất metadata, chuyển đổi xám/màu
│   ├── visualization.py        # Vẽ Histogram RGB/Xám, Before/After, Bar chart Matplotlib
│   ├── history.py              # Quản lý trạng thái tuần tự, Undo, Redo, Reset, History log
│   └── download.py             # Đóng gói dữ liệu tải về định dạng PNG / JPG
│
├── sample_images/              # Bộ ảnh mẫu phục vụ thử nghiệm và thuyết trình
│   ├── README.md
│   ├── low_contrast.jpg        # Tương phản rất thấp (sương mù, mờ đục)
│   ├── dark_image.jpg          # Thiếu sáng (Underexposed)
│   ├── bright_image.jpg        # Dư sáng (Overexposed)
│   ├── uneven_lighting.jpg     # Chiếu sáng không đồng đều (minh họa ưu thế CLAHE)
│   └── sample_grayscale.png    # Ảnh mức xám chuẩn 2D
│
└── results/                    # Thư mục lưu trữ kết quả thực nghiệm
    └── README.md
```

---

## 6. CÀI ĐẶT & CHẠY PROJECT TRONG VISUAL STUDIO CODE

### Bước 1: Mở Terminal trong Visual Studio Code

Mở Visual Studio Code, chọn `Terminal -> New Terminal` (phím tắt `Ctrl + ~`).

### Bước 2: Di chuyển vào thư mục dự án

```bash
cd histogram-image-enhancement
```

### Bước 3: Tạo môi trường ảo Python

```bash
python -m venv .venv
```

### Bước 4: Kích hoạt môi trường ảo (Windows)

```powershell
.venv\Scripts\activate
```

### Bước 5: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### Bước 6: Khởi chạy ứng dụng Streamlit

```bash
streamlit run app.py
```
*(Hoặc chạy lệnh: `python -m streamlit run app.py` hoặc nhấp đúp file `run.bat`)*

Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 7. HƯỚNG DẪN SỬ DỤNG GIAO DIỆN

### 7.1 Tải ảnh lên
- Tại **Sidebar** bên trái, chọn tệp tin ảnh từ máy tính hoặc chọn một trong các **ảnh mẫu có sẵn**.
- Hệ thống tự động ghi nhận ảnh vào `original_image` và khởi tạo `current_image`.

### 7.2 Trung tâm Xử lý & Cải thiện ảnh (Chức năng 2)
- Màn hình hiển thị song song **Ảnh gốc (Original)** và **Ảnh đang chỉnh sửa (Current)**.
- Bên dưới là các tab công cụ chuyên biệt:
  - **Tab 1 - Ánh sáng & Tương phản:** Kéo thanh trượt Brightness, Contrast, Gamma, Exposure, Highlights, Shadows. Nhấn **"Xem trước"** để quan sát thử, nhấn **"Áp dụng"** để cập nhật vào `current_image`.
  - **Tab 2 - Histogram Equalization & CLAHE:** Cân bằng toàn cục bảo toàn màu qua YCrCb hoặc cân bằng cục bộ CLAHE với Clip Limit và Tile Grid Size.
  - **Tab 3 - Khử nhiễu:** Lựa chọn bộ lọc Bilateral, Gaussian, Median, Non-Local Means theo 3 mức độ Nhẹ / Vừa / Mạnh.
  - **Tab 4 - Làm nét & Chi tiết:** Làm nét Unsharp Mask, tăng cường đường biên Sobel, phục hồi ảnh mờ Richardson-Lucy/Wiener.
  - **Tab 5 - Màu sắc & Cân bằng trắng:** Chỉnh độ bão hòa Saturation, độ lệch sắc Hue, nhiệt độ màu Temperature, cân bằng trắng Gray World, Sepia, Grayscale.
  - **Tab 6 - Tự động cải thiện:** Nhấn 1 nút để hệ thống chẩn đoán và tự động chạy phác đồ tối ưu.
  - **Tab 7 - Lịch sử chỉnh sửa:** Xem lại danh sách toàn bộ các bước đã làm, chọn xem lại hoặc khôi phục về bất kỳ bước nào trong quá khứ.
- **Thanh điều khiển:**
  - `↩ Hoàn tác (Undo)`: Quay lại bước trước đó.
  - `↪ Làm lại (Redo)`: Khôi phục bước vừa hoàn tác.
  - `↻ Khôi phục gốc (Reset)`: Đặt lại trạng thái về ảnh gốc ban đầu.
  - `📥 Tải ảnh hiện tại`: Xuất file PNG hoặc JPG.

### 7.3 Các chức năng phân tích & đối chiếu khác
- **3. Histogram & phân tích:** Khảo sát biểu đồ histogram của `current_image` theo các kênh màu và nhận chẩn đoán tự động.
- **4. So sánh thuật toán:** Chọn nhiều phương pháp để chạy **độc lập từ ảnh gốc**, theo dõi bảng đo thời gian xử lý (ms) và các chỉ số so sánh công bằng.
- **5. Đánh giá kết quả:** Bảng chỉ số đối chiếu `original_image` vs `current_image` (Mean, Std, Entropy, PSNR, SSIM) kèm biểu đồ cột Matplotlib.
- **6. So sánh Before / After:** Trực quan hóa ảnh trước và sau cùng 2 biểu đồ histogram đối xứng.
- **7. Demo tự động & Thực nghiệm:** Trình diễn toàn bộ chuỗi thuật toán liên tiếp phục vụ báo cáo trên lớp.
- **8. Cơ sở lý thuyết:** Cung cấp đầy đủ công thức toán học và giải thích chuyên sâu 14 mục lý thuyết.

---

## 8. CÁC THUẬT TOÁN KHOA HỌC

1. **Histogram Equalization qua YCrCb:**
   $$Y_{new} = \text{round}\left( \frac{CDF(Y) - CDF_{min}}{M \times N - CDF_{min}} \times 255 \right)$$
   Giữ nguyên $Cr, Cb$ để bảo toàn 100% sắc độ tự nhiên của vật thể.
2. **CLAHE:** Chia lưới $M \times N$ ô, cắt đỉnh histogram tại ngưỡng `clipLimit`, phân phối lại pixel dư thừa và nội suy song tuyến (Bilinear Interpolation).
3. **Gamma Correction:** $I' = 255 \times (I / 255)^\gamma$. Kéo sáng vùng tối phi tuyến tính mà không làm cháy sáng vùng highlights.
4. **Lọc song phương (Bilateral Filter):** Kết hợp nhân không gian Gauss $g_s$ và nhân khoảng cách cường độ $f_r$ để làm mịn bề mặt mà giữ nguyên cạnh sắc nét.
5. **Cân bằng trắng (Gray World):** $\mu_{gray} = (\mu_R + \mu_G + \mu_B) / 3$; chuẩn hóa $R' = R \times (\mu_{gray} / \mu_R)$.

---

## 9. KẾT QUẢ THỰC NGHIỆM

| Trường hợp thử nghiệm | Trạng thái ban đầu | Chuỗi xử lý tối ưu | Kết quả đạt được |
| :--- | :--- | :--- | :--- |
| **Ảnh tương phản thấp** (`low_contrast.jpg`) | Mức xám co cụm [116 - 163], Std = 13.08 | `Contrast x1.2` ➜ `CLAHE (Clip 2.0)` ➜ `Sharpen 35` | Std tăng lên > 35, chi tiết nổi khối sắc nét |
| **Ảnh thiếu sáng** (`dark_image.jpg`) | Mean < 60, histogram lệch trái | `Brightness +30` ➜ `Gamma 0.85` ➜ `Bilateral Filter` | Phục hồi rõ chi tiết vùng tối, hạn chế nhiễu hạt |
| **Chiếu sáng không đều** (`uneven_lighting.jpg`) | Nửa trái quá tối, nửa phải quá sáng | `CLAHE (Grid 8x8, Clip 2.0)` ➜ `Sharpen 30` | Cả hai vùng đều được cân bằng rõ rệt |
| **Ảnh ám màu** | Chênh lệch lớn giữa các kênh R, G, B | `White Balance (Gray World)` ➜ `Saturation 110%` | Tông màu trở về tự nhiên, trung thực |

---

## 10. ĐÁNH GIÁ CHẤT LƯỢNG

- **Độ sáng (Mean):** Đo lường mức sáng tổng thể; hệ thống giúp đưa mức sáng về dải thoải mái cho thị giác người (110 – 140).
- **Độ tương phản (Std):** Tăng độ lệch chuẩn phản ánh ranh giới giữa sáng và tối rõ ràng hơn.
- **Entropy Shannon:** Đo lường lượng thông tin chi tiết; Entropy tăng chứng tỏ các chi tiết ẩn trong mức xám được khai phóng.
- **PSNR & SSIM:** Theo dõi mức độ biến đổi tín hiệu và đảm bảo tính toàn vẹn cấu trúc vật thể không bị méo mó.

---

## 11. HẠN CHẾ CỦA PHIÊN BẢN HIỆN TẠI

- Phục hồi ảnh mờ nặng ở mức cơ bản do phụ thuộc vào giả định hàm truyền điểm PSF (ảnh mất thông tin quá nặng không thể phục hồi 100%).
- Khử nhiễu Non-Local Means trên ảnh có độ phân giải rất lớn (> 4K) có thể mất vài giây xử lý trên CPU.

---

## 12. HƯỚNG PHÁT TRIỂN

Cấu trúc module hóa độc lập của hệ thống cho phép tích hợp các công nghệ nâng cao trong tương lai:
1. **Edge Detection:** Phát hiện biên cạnh tự động bằng toán tử Canny, Laplacian of Gaussian, Sobel.
2. **Super Resolution & AI Denoising:** Ứng dụng mô hình Deep Learning (ESRGAN, DnCNN) để tăng độ phân giải siêu nét.
3. **Face & Portrait Enhancement:** Nhận diện khuôn mặt và làm đẹp tự động.
