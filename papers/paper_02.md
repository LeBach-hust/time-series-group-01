**Tên bài báo:** Enhanced TSMixer Model for the Prediction and Control of Particulate Matter (E-TSMixer)

**Tác giả:** C. Yang, H. Li, Y. Ma, Y. Huang, X. Chu — Đại học Thâm Quyến (Shenzhen University)

**Năm — Nơi công bố:** 2025 — Sustainability (MDPI), Vol. 17(7), bài số 2933

**Xếp hạng:** Q1 (Scimago, nhóm ngành Geography, Planning & Development)

**Link / DOI:** DOI 10.3390/su17072933  •  mdpi.com/2071-1050/17/7/2933 (truy cập mở)

**Đa biến → một chiều ra?:** Có — hợp nhất 14 thông số chất lượng không khí + 12 thông số giao thông → một chiều ra (PM2.5).

**① Vấn đề nghiên cứu:** Dự báo nồng độ bụi mịn PM2.5 đô thị để điều khiển xe phun nước chống bụi theo thời gian thực. Khó khăn: dữ liệu đa nguồn (khí tượng + giao thông), phi tuyến, tần suất phút; và quan trọng nhất là phải bắt được các ĐỈNH vượt ngưỡng quy chuẩn 35 µg/m³ để cảnh báo sớm, chứ không chỉ giảm sai số trung bình. Mô hình thống kê (ARIMA) yếu với phi tuyến/đa biến; mô hình vật lý chậm và cần nhiều tham số; LSTM/Transformer chính xác hơn nhưng nặng tính toán.

**② Ý tưởng chính:** Lấy TSMixer (kiến trúc MLP-mixing nhẹ, tách riêng việc trộn theo chiều thời gian và chiều đặc trưng — nhanh hơn RNN/Transformer) làm lõi, rồi cải tiến HAI điểm cho hợp bài toán: (1) ép đầu ra đa biến về MỘT chiều mục tiêu; (2) sửa HÀM MẤT MÁT để ưu tiên bắt đỉnh thay vì chỉ tối ưu sai số trung bình.

**③ Mô hình đề xuất:** • Lõi TSMixer: xếp chồng các Mixer Layer gồm time-mixing (lớp FC theo trục thời gian + ReLU + residual) và feature-mixing (lớp FC theo trục đặc trưng), rồi lớp Temporal Projection (FC) ánh xạ độ dài đầu vào sang độ dài dự báo.
• Cải tiến 1 — Fully-connected output layer: thêm lớp FC ở cuối để ánh xạ đầu ra đa chiều về MỘT chiều (PM2.5) → đúng dạng đa-biến-vào → một-chiều-ra của bài toán nhóm.
• Cải tiến 2 — Loss bất đối xứng (asymmetric penalty): TL = Σ(ŷ≥y)(ŷ−y)² + λ·Σ(ŷ<y)(ŷ−y)². Khi dự báo THẤP HƠN thực tế (bỏ lỡ đỉnh) thì bị phạt nặng gấp λ lần → mô hình thiên về dự báo "thà cao hơn còn hơn bỏ sót đỉnh".
• Nhập liệu: cửa sổ 1440 phút (1 ngày), làm mượt 15 phút, dự báo 6h/10h tới.

**④ Kết quả chính:** So với Transformer, TSMixer giảm ~19.9% RMSE, ~21.7% MAE, ~22% SMAPE, đồng thời giảm ~90% thời gian train và ~40% thời gian suy luận. TSMixer gốc: RMSE 4.02, MAE 3.02, SMAPE 13.41 — nhưng bỏ lỡ khoảng 15% số ca vượt ngưỡng 35 µg/m³. E-TSMixer: sai số trung bình hơi cao hơn (RMSE 4.57, MAE 3.42, SMAPE 15.17) NHƯNG bắt đỉnh tốt hơn rõ rệt (giảm mạnh false-negative tại các đỉnh vượt ngưỡng), trong khi chi phí gần như không tăng (train 48.76 s/epoch so với 25.97 s; suy luận 2.00 ms so với 1.98 ms mỗi điểm). Thước đo: MAE / RMSE / SMAPE.

**⑤ Điểm mạnh / Hạn chế:** MẠNH: (1) Backbone MLP nhẹ, nhanh, dễ song song hóa. (2) Loss bất đối xứng giải đúng nhu cầu thực tế (bắt đỉnh, cảnh báo sớm). (3) Hợp nhất đa nguồn dữ liệu và xuất một chiều — đúng khuôn đa-biến-vào → một-chiều-ra. (4) Chi phí tính toán tăng không đáng kể.
HẠN CHẾ: (1) Loss bất đối xứng đánh đổi sai số TRUNG BÌNH cao hơn để đổi lấy khả năng bắt đỉnh — không phải lúc nào cũng mong muốn. (2) λ là siêu tham số phải dò, và phụ thuộc ngưỡng quy chuẩn cụ thể (35 µg/m³). (3) Loss chỉ phạt theo DẤU sai lệch (cao/thấp), CHƯA phạt theo cấu trúc/tương quan thời gian. (4) TSMixer xử lý quan hệ giữa các biến bằng FC tuyến tính, không học tương quan biến một cách tường minh như attention.

**⑥ Khả năng áp dụng cho nhóm:** Bài tham chiếu TRỰC TIẾP cho hướng cải tiến LOSS của nhóm: cách họ thiết kế penalty bất đối xứng chứng minh việc "đưa mục tiêu nghiệp vụ vào hàm mất mát" là khả thi và rẻ. Nhóm có thể đi xa hơn: thay vì phạt theo dấu (cao/thấp), thiết kế loss phạt sai lệch CẤU TRÚC TƯƠNG QUAN (vd khác biệt giữa ma trận tương quan thời gian của dự báo và của thực tế) — đây là điểm khác biệt rõ so với asymmetric penalty. Lớp FC ép-một-chiều của họ cũng đúng khuôn đầu ra bài toán nhóm. Lưu ý: dữ liệu PM2.5 ≠ điện, nhưng cơ chế loss có thể chuyển giao.

