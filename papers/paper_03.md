**Tên bài báo:** Enhanced forecasting of shipboard electrical power demand using multivariate input and variational mode decomposition with mode selection (VMDMS)

**Tác giả:** P. Fazzini, G. La Tona, M. Diez, M. C. Di Piazza — CNR-INM, Italy

**Năm — Nơi công bố:** 2025 — Scientific Reports (Nature Portfolio)

**Xếp hạng:** Q1 (Scimago, nhóm ngành Multidisciplinary)

**Link / DOI:** DOI 10.1038/s41598-025-06153-z  •  nature.com/articles/s41598-025-06153-z (truy cập mở)

**Đa biến → một chiều ra?:** Có — nhiều chuỗi đầu vào (đa kênh) → dự báo nhu cầu điện trên tàu.

**① Vấn đề nghiên cứu:** Dự báo chính xác nhu cầu điện trên tàu khách cỡ lớn nhằm tối ưu Hệ thống Quản lý Năng lượng (EMS) — yếu tố then chốt để vận hành lưới điện trên tàu hiệu quả và sinh lời. Chuỗi tải điện biến động mạnh, phi tuyến, nhiều biến tương tác. Phép phân rã VMD (Variational Mode Decomposition) trước đây chủ yếu dùng cho dữ liệu ĐƠN biến; mở rộng sang đa biến gặp khó vì phải xác định mode nào ở kênh nào là thật sự hữu ích.

**② Ý tưởng chính:** Phương pháp lai: phân rã chuỗi ĐA BIẾN bằng một biến thể VMD mới — VMDMS (VMD with Mode Selection) — rồi đưa các mode đã được CHỌN LỌC vào mạng LSTM để dự báo. Mấu chốt là quá trình chọn mode: tự động phát hiện các mode (thành phần dải tần) XUYÊN KÊNH có tác dụng cộng hưởng làm tăng độ chính xác, đồng thời loại bỏ mode nhiễu/vô ích.

**③ Mô hình đề xuất:** • VMDMS: phân rã mỗi kênh đầu vào thành các Intrinsic Mode Function (IMF — thành phần dải tần); cơ chế chọn mode xác định tập mode (xuyên các kênh) phối hợp tốt nhất cho dự báo, không áp đặt giả định hạn chế lên dữ liệu (nhờ đó mở rộng được VMD sang đa biến).
• Bộ dự báo: mạng LSTM nhận các mode đã chọn (đa biến đã phân rã) → sinh dự báo nhu cầu điện đa bước.
• Kiểm chứng trên dữ liệu thực thu từ một tàu khách cỡ lớn.

**④ Kết quả chính:** Thực nghiệm xác nhận hiệu quả của VMDMS + LSTM, mở rộng được VMD sang dự báo đa biến mà không cần giả định ràng buộc lên dữ liệu; đóng góp vào việc tối ưu phương pháp phân rã cho mô hình dự báo trong quản lý năng lượng. (Nghiên cứu cùng nhóm trước đó cho thấy phân rã VMD trước khi dự báo có thể giảm tới ~66% sai số so với đưa thẳng tín hiệu vào mô hình.) Đánh giá trên bài toán dự báo tải điện đa bước.

**⑤ Điểm mạnh / Hạn chế:** MẠNH: (1) Cơ chế CHỌN MODE xuyên kênh — lọc thành phần đầu vào hữu ích, giảm nhiễu trước khi mô hình hóa. (2) Mở rộng VMD sang đa biến mà không cần giả định ràng buộc. (3) Kiểm chứng trên dữ liệu tải điện THỰC — gần với bài toán điện năng của nhóm.
HẠN CHẾ: (1) Bộ dự báo dùng LSTM (xử lý tuần tự, chậm, khó song song) thay vì kiến trúc hiện đại hơn. (2) Quy trình hai pha (phân rã → dự báo) làm pipeline phức tạp, phát sinh thêm siêu tham số (số mode, ngưỡng chọn). (3) VMD tốn tính toán; chi phí có thể lớn với chuỗi dài hoặc nhiều kênh.

**⑥ Khả năng áp dụng cho nhóm:** Bài tham chiếu cho hướng XỬ LÝ BIẾN NGOẠI SINH của nhóm: ý tưởng CHỌN LỌC thành phần/kênh hữu ích (mode selection) song song với ý tưởng "lọc biến ngoại sinh theo tương quan/độ trễ" của nhóm — cùng triết lý "chỉ giữ tín hiệu có ích trước khi đưa vào mô hình". Đây cũng là bài điện năng thực, hợp bối cảnh ứng dụng. Khác biệt nhóm cần nêu rõ: thay vì lọc theo dải tần (VMD), nhóm lọc theo TƯƠNG QUAN CHÉO + ĐỘ TRỄ giữa ngoại sinh và mục tiêu, và nhúng thẳng vào TimeXer thay vì pipeline VMD+LSTM rời rạc.

