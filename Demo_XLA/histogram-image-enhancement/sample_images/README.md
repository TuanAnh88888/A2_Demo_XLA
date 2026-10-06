# Thư mục Ảnh Mẫu (Sample Images)

Thư mục này chứa các ảnh thử nghiệm đại diện cho các tình huống điển hình trong xử lý ảnh và tăng cường chất lượng ảnh:

1. **`low_contrast.jpg`**: Ảnh có độ tương phản rất thấp (sương mù / mờ nhạt, dải mức xám bị nén chặt ở khoảng giữa [100 - 155]). Thích hợp kiểm tra khả năng kéo giãn dải động của Histogram Equalization và CLAHE.
2. **`dark_image.jpg`**: Ảnh chụp thiếu sáng (Underexposed, mức xám tập trung về phía 0). Thích hợp thử nghiệm tăng Brightness, Histogram Equalization và phục hồi chi tiết bóng tối.
3. **`bright_image.jpg`**: Ảnh chụp dư sáng (Overexposed, mức xám tập trung về phía 255). Thích hợp thử nghiệm giảm Brightness hoặc giảm tương phản để khôi phục vùng quá sáng.
4. **`uneven_lighting.jpg`**: Ảnh có độ chiếu sáng không đồng đều (nửa trái tối sâu, nửa phải sáng mạnh). Là ví dụ kinh điển chứng minh sự vượt trội của **CLAHE** (cân bằng cục bộ) so với **Histogram Equalization** (cân bằng toàn cục).
5. **`sample_grayscale.png`**: Ảnh mức xám chuẩn 1 kênh đơn để khảo sát trực tiếp các thuật toán xử lý trên ma trận 2D.
