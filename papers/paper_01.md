**Chủ đề: Cải tiến mô hình TimeXer (hàm mất mát + xử lý biến ngoại sinh) — dự báo chuỗi thời gian đa biến, đầu ra một chiều**

## BÀI 1 — Mô hình lõi

**Tên bài báo:** TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

**Tác giả:** Y. Wang, H. Wu, J. Dong, G. Qin, H. Zhang, Y. Liu, Y. Qiu, J. Wang, M. Long — THUML, Đại học Thanh Hoa

**Năm — Nơi công bố:** 2024 — NeurIPS 2024 (hội nghị hàng đầu, hạng A* về Machine Learning)

**Xếp hạng:** Hội nghị A* (không xếp quartile theo Scimago/JCR — quartile chỉ dành cho tạp chí)

**Link / DOI:** arXiv:2402.19072  •  github.com/thuml/TimeXer

**Đa biến → một chiều ra?:** Có — chạy cả hai chế độ: dự báo có biến ngoại sinh và dự báo đa biến (qua channel independence).

**① Vấn đề nghiên cứu:** Trong thực tế, một hệ thống được ghi thành nhiều biến; biến ngoại sinh (thời tiết, sự kiện…) mang thông tin bổ trợ quý cho biến mục tiêu (nội sinh). Hai cách làm cũ đều chưa tối ưu: dự báo đa biến "thuần" coi mọi biến như nhau (dễ đưa nhiễu vào), còn dự báo đơn biến lại bỏ qua ngoại sinh. Cách nhập ngoại sinh phổ biến là cộng/nối thẳng vào nội sinh — thô và không khai thác đúng vai trò của chúng. Bài đặt câu hỏi: làm sao để Transformer tiếp nhận thông tin ngoại sinh có chọn lọc mà không phải đại tu kiến trúc.

**② Ý tưởng chính:** Trao cho Transformer chuẩn khả năng dung hòa nội sinh và ngoại sinh CHỈ bằng cách thiết kế khéo lớp embedding, KHÔNG đổi lõi attention. Mấu chốt: dùng độ chi tiết biểu diễn khác nhau cho hai loại biến — biến nội sinh (mục tiêu) biểu diễn ở mức PATCH (đoạn) để giữ chi tiết thời gian; mỗi biến ngoại sinh được nén thành MỘT token mức biến (variate) để tránh đưa nhiễu chi tiết vào. Một "global endogenous token" đóng vai cầu nối, mang thông tin nhân quả từ ngoại sinh về cho chuỗi nội sinh.

**③ Mô hình đề xuất:** • Embedding: chuỗi nội sinh được cắt thành N patch → N patch token + 1 global token (học được); mỗi chuỗi ngoại sinh → 1 variate token (embedding toàn chuỗi).
• Hai cơ chế attention chạy đồng thời trong mỗi block:
   (a) Patch-wise self-attention giữa các patch token (gồm global token) → học phụ thuộc thời gian nội tại của biến mục tiêu.
   (b) Variate-wise cross-attention: global token làm query, các variate token ngoại sinh làm key/value → "hút" thông tin ngoại sinh liên quan về global token, sau đó global token lan tỏa lại cho các patch.
• Tổng quát hóa sang đa biến bằng channel independence: lần lượt lấy từng biến làm nội sinh, các biến còn lại làm ngoại sinh, chia sẻ trọng số self-/cross-attention cho mọi biến.

**④ Kết quả chính:** Đạt SOTA nhất quán trên 12 benchmark thực. Mạnh ở cả hai chế độ: (i) dự báo ngắn hạn có ngoại sinh — gồm bộ EPF (giá điện ngắn hạn ở 5 thị trường, 6 năm mỗi thị trường), nơi biến mục tiêu tương quan mạnh với 2 biến ngoại sinh; (ii) dự báo dài hạn đa biến. Vượt iTransformer, Crossformer, PatchTST, DLinear, TimesNet… Bài chỉ ra: Crossformer mô hình mọi biến ở mức chi tiết lại kém vì đưa nhiễu không cần thiết; iTransformer thiếu attention theo thời gian nên yếu ở phụ thuộc thời gian — TimeXer khắc phục cả hai bằng cách kết hợp patch (thời gian) + variate (biến).

**⑤ Điểm mạnh / Hạn chế:** MẠNH: (1) Đơn giản, không sửa kiến trúc Transformer → dễ triển khai, dễ mở rộng. (2) Kết hợp được CẢ phụ thuộc thời gian (patch self-attention) lẫn tương quan biến (variate cross-attention). (3) Dùng chung cho cả hai paradigm (ngoại sinh & đa biến). (4) Có mã nguồn chính thức trong khung Time-Series-Library → dễ tái lập và benchmark.
HẠN CHẾ: (1) Phải CHỌN biến ngoại sinh thủ công — đưa vào biến không phù hợp sẽ làm giảm độ chính xác. (2) Mô hình tương quan theo chiến lược HAI BƯỚC (thời gian trước, biến sau) có thể gây nhiễu lẫn nhau giữa hai bước. (3) Cross-attention "thuần" chưa tính đến ĐỘ TRỄ và tương quan ĐỘNG giữa ngoại sinh và nội sinh (vd nhiệt độ ảnh hưởng tải điện sau vài giờ). (4) Mỗi biến ngoại sinh bị nén thành 1 token toàn cục → mất thông tin biến thiên theo thời gian của chính biến ngoại sinh.

**⑥ Khả năng áp dụng cho nhóm:** Đây là MÔ HÌNH LÕI nhóm chọn để cải tiến và đem giải bài toán thực hành (đa-biến-vào → một-chiều-ra: lấy chiều mục tiêu làm nội sinh, các biến còn lại làm ngoại sinh). Hai hạn chế (1) và (3) ở mục ⑤ chính là cửa cho hướng XỬ LÝ BIẾN NGOẠI SINH (tự động lọc biến theo tương quan-trễ, hoặc embedding nhận biết độ trễ). Việc mô hình chỉ tối ưu sai số điểm là cửa cho hướng LOSS (thêm thành phần phạt theo cấu trúc/tương quan). Có sẵn bộ EPF và mã nguồn trong TSLib nên dựng baseline và cải tiến khá nhanh.

