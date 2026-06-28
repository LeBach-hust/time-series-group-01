# Nghiên cứu cải tiến mô hình TimeXer dựa trên mức độ tương quan thời gian

## 👥 Tên nhóm và thành viên
**Khoa Toán - Tin, Đại học Bách Khoa Hà Nội**
* **Lê Việt Bách** - 20237302
* **Nguyễn Văn Đông** - 20237309
* **Đào Việt Trung** - 20237396
* **Giảng viên hướng dẫn:** PGS.TS. Nguyễn Thị Ngọc Anh

---

## 🎯 Chủ đề nghiên cứu
**Nghiên cứu cải tiến mô hình TimeXer dựa trên mức độ tương quan thời gian và ứng dụng trong dự báo lượng tiêu thụ điện năng.**

Đồ án tập trung giải quyết bài toán dự báo phụ tải điện năng bằng cách đề xuất hai cải tiến trực giao cho kiến trúc học sâu TimeXer: (1) thiết kế hàm mất mát có trọng số theo mức độ tương quan thời gian và (2) xây dựng cơ chế biến ngoại sinh có khả năng nhận biết độ trễ dựa trên tương quan phi tuyến.

---

## 📄 Ba bài báo tham khảo nền tảng
Quá trình nghiên cứu và xây dựng mô hình được phát triển dựa trên nền tảng lý thuyết từ 3 công trình tiêu biểu sau:

1. **Bài 1 (Mô hình lõi): TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables**
   * **Tác giả:** Y. Wang, H. Wu, J. Dong, G. Qin, H. Zhang, Y. Liu, Y. Qiu, J. Wang, M. Long (THUML, Đại học Thanh Hoa).
   * **Nơi công bố:** NeurIPS 2024 (Hội nghị hạng A* về Machine Learning).

2. **Bài 2 (Cải tiến hàm mất mát): Enhanced TSMixer Model for the Prediction and Control of Particulate Matter (E-TSMixer)**
   * **Tác giả:** C. Yang, H. Li, Y. Ma, Y. Huang, X. Chu (Đại học Thâm Quyến).
   * **Nơi công bố:** Sustainability (MDPI), Vol. 17(7), bài số 2933, năm 2025 (Xếp hạng Q1).

3. **Bài 3 (Xử lý / chọn lọc đầu vào): Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection (VMDMS)**
   * **Tác giả:** P. Fazzini, G. La Tona, M. Diez, M. C. Di Piazza (CNR-INM, Italy).
   * **Nơi công bố:** Scientific Reports (Nature Portfolio), năm 2025 (Xếp hạng Q1).

---

## 📊 Mô tả bộ dữ liệu
* **Nguồn dữ liệu:** Dữ liệu nhu cầu điện năng (ENTSO-E) và thời tiết (OpenWeather) của **Tây Ban Nha (giai đoạn 2015-2018)**.
* **Quy mô:** Dữ liệu được ghi nhận với độ phân giải 1 giờ/lần (tổng cộng 35.064 quan sát).
* **Biến mục tiêu (Endogenous):** Tổng phụ tải điện thực tế của hệ thống (MW).
* **Biến ngoại sinh (Exogenous):** Nhiệt độ (°C), độ ẩm, áp suất, tốc độ gió, độ che phủ mây.

---

## ⚙️ Phương pháp thực hiện
Dự án xây dựng mô hình dự báo dựa trên kiến trúc **TimeXer** kết hợp cùng 2 hướng cải tiến độc lập nhưng bổ trợ cho nhau:

1. **Hàm mất mát có trọng số tương quan (Time-Correlation Weighted Loss):** Thay thế hàm MSE truyền thống bằng một hàm mất mát sử dụng trọng số nghịch phương sai ($w_h^{invvar}$). Trọng số này được suy ra từ hiệp phương sai sai số dự báo, có tính chất giảm đơn điệu theo tầm nhìn dự báo (horizon), giúp mô hình tập trung tối ưu các bước dự báo gần có độ tin cậy cao.
2. **Cơ chế ngoại sinh XCorr-Lag Exogenous:**
   * **Pha 1 (Lập hồ sơ):** Tự động khám phá độ trễ và mức độ liên quan phi tuyến giữa từng biến ngoại sinh với phụ tải bằng **Tương quan khoảng cách (distance correlation - dCor)**.
   * **Pha 2 (Căn trễ & Gating):** Áp dụng phép căn pha thông tin (nhân quả) và sử dụng hệ thống **cổng liên quan học được (learnable gating)**. Đặc tính này giúp mô hình khôi phục được các mối quan hệ phi tuyến phức tạp (như ảnh hưởng hình chữ U của nhiệt độ) mà hệ số Pearson tuyến tính thường bỏ sót.

---

## 🏆 Bảng kết quả mô hình
Cấu hình huấn luyện: Cửa sổ nhìn lại $L=168$, Tầm dự báo $H=24$, Kích thước đoạn $P=24$.  
*Lưu ý: Để đảm bảo tính khách quan, các cấu hình đều được đánh giá chéo trên 3 hạt giống (seeds) khác nhau. Kết quả hiển thị là trung bình $\pm$ độ lệch chuẩn.*

| Cơ chế ngoại sinh | Hàm loss | MAE (chuẩn hóa) | MSE (chuẩn hóa) | RMSE (chuẩn hóa) |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` (TimeXer gốc) | `mse` | $0.2840 \pm 0.0064$ | $0.1732 \pm 0.0044$ | $0.4161 \pm 0.0052$ |
| `baseline` | `weighted` | $0.2651 \pm 0.0064$ | $0.1598 \pm 0.0034$ | $0.3997 \pm 0.0043$ |
| `xcorr_lag` | `mse` | $0.2619 \pm 0.0124$ | $0.1629 \pm 0.0081$ | $0.4035 \pm 0.0100$ |
| **`xcorr_lag` (Đề xuất)** | **`weighted` (Đề xuất)** | **$0.2542 \pm 0.0053$** | **$0.1556 \pm 0.0041$** | **$0.3945 \pm 0.0052$** |

---

## 💻 Hướng dẫn chạy mã nguồn (Môi trường Google Colab)

Dự án được phân tách thành các module chuẩn mực và thiết kế để chạy mượt mà trên môi trường Google Colab có hỗ trợ GPU. Dưới đây là các bước thực thi chi tiết:

### Bước 1: Chuẩn bị môi trường và Dữ liệu
1. Tải toàn bộ mã nguồn (`features.py`, `data_loader.py`, `models.py`, `evaluation.py`, `requirements.txt`) lên một thư mục trên Google Drive của bạn.
2. Tạo thư mục chứa dữ liệu tại đường dẫn: `MyDrive/Time_Series_Group_1/Data/Processed/` và tải lên 4 file dữ liệu cần thiết:
   * `clean_hourly.csv`
   * `train.csv`
   * `val.csv`
   * `test.csv`
   * *(Tùy chọn)* `loss_weights.npz` (nếu đã lưu trọng số loss tính sẵn).

### Bước 2: Mount Google Drive trên Colab
Tạo một file Notebook mới (`.ipynb`) nằm cùng cấp với thư mục mã nguồn và chạy cell sau để kết nối Drive, đồng thời chuyển thư mục làm việc (workspace):

```python
from google.colab import drive
drive.mount('/content/drive')

# Di chuyển vào thư mục chứa code của bạn (thay đổi đường dẫn nếu cần)
%cd /content/drive/MyDrive/Time_Series_Group_1/

# 4. Huấn luyện mô hình và chạy bảng đánh giá Ablation 2x2
python evaluation.py
