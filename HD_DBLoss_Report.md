# Báo Cáo Nghiên Cứu & Phát Triển Hàm Mất Mát HD-DBLoss
**(Step-wise Empirical Variance Weighting for Decomposition-Based Time Series Forecasting)**

---

## 1. Động Lực Nghiên Cứu & Lý Do Loại Bỏ Heuristic Theo Horizon $H$

### 1.1. Hạn chế của cách tiếp cận Heuristic theo $H$ trước đây
Trong các phiên bản thử nghiệm trước (V1–V9), chúng ta đã sử dụng các công thức tĩnh phụ thuộc trực tiếp vào độ dài cửa sổ dự báo $H$:
- Gán cứng quy luật phương sai tĩnh $s_h = 1 + \tau \frac{h-1}{H-1}$ mà không đo từ dữ liệu thực tế.
- Dùng hàm `_horizon_params(H)` để thay đổi thủ công $\alpha(H)$, $\beta(H)$, $\gamma(H)$ theo $H$. Cách làm này mang tính **heuristic (chỉnh tay siêu tham số theo $H$)**, làm thay đổi cả bộ lọc phân rã EMA và tỷ trọng nhánh $\beta$ gốc của `DBLoss`, khiến kết quả phụ thuộc giả tạo vào cấu hình của `PatchTST` trên từng horizon.
- Ngoài ra, nếu chỉ nhân một trọng số cố định không phụ thuộc dữ liệu hoặc thay đổi scale tổng thể, bộ tối ưu **Adam** (với cơ chế chia cho căn bậc hai moment bậc hai $\sqrt{\hat{v}_t}$) sẽ tự động chuẩn hóa lại biên độ đạo hàm của từng tham số.

### 1.2. Nguyên lý Trọng số theo Phương sai Thực nghiệm từng Bước Thời gian (Step-wise Empirical Variance)
Để đảm bảo tính trung thực khoa học và công bằng tuyệt đối khi so sánh với `DBLoss` gốc:
1. **Giữ cố định 100% siêu tham số nền của `DBLoss`**: $\alpha = 0.2$, $\beta = 0.5$ cho mọi horizon $H \in \{96, 192, 336, 720\}$.
2. **Đo trực tiếp phương sai thực nghiệm tại từng bước $h \in \{1, \dots, H\}$ từ dữ liệu huấn luyện**:
   - Thay vì đoán trước phương sai bước $h$ bằng hàm toán học tĩnh, ta theo dõi sai số thực tế của mô hình tại từng bước dự báo $h$ thông qua bộ nhớ đệm EMA (`running_var_s` và `running_var_t`, độ dài $H$).
   - Vì trọng số $w_h$ thay đổi tương đối giữa các bước $h=1\dots H$ dựa trên phân phối sai số thực tế của dữ liệu và tất cả các bước $h$ cùng đi qua chung các tầng chiếu của mạng (shared weights), sự thay đổi tỷ trọng tương đối giữa các bước $h$ trực tiếp định hướng lại tổng vector gradient $\nabla_\theta \mathcal{L} = \sum_{h=1}^H w_h \nabla_\theta \ell_h$ mà **không bị Adam triệt tiêu**.

---

## 2. Công Thức Toán Học Của HD-DBLoss (Empirical Variance Weighting)

### Bước 1: Phân rã chuỗi bằng EMA như `DBLoss` gốc ($\alpha = 0.2$)
Với dự đoán $\hat{Y} \in \mathbb{R}^{B \times H \times C}$ và nhãn thực tế $Y \in \mathbb{R}^{B \times H \times C}$:
$$\hat{S}, \hat{T} = \text{DECOMP}_{\alpha=0.2}(\hat{Y}), \quad S, T = \text{DECOMP}_{\alpha=0.2}(Y)$$
Tính sai số từng điểm tại mỗi bước thời gian $h \in \{1, \dots, H\}$:
$$e_{S}(b, h, c) = \left(\hat{S}_{b, h, c} - S_{b, h, c}\right)^2, \quad e_{T}(b, h, c) = \left|\hat{T}_{b, h, c} - T_{b, h, c}\right|$$

### Bước 2: Cập nhật Phương sai Thực nghiệm bằng EMA Buffer (`momentum = 0.95`)
Trong mỗi batch huấn luyện (`self.training == True`), tính sai số trung bình tại từng bước $h$ (gộp qua chiều Batch $B$ và Channel $C$, không tính đạo hàm `torch.no_grad()`):
$$\hat{\sigma}_{S, \text{batch}}^2(h) = \frac{1}{B C} \sum_{b=1}^{B} \sum_{c=1}^{C} e_S(b, h, c), \quad \hat{\sigma}_{T, \text{batch}}(h) = \frac{1}{B C} \sum_{b=1}^{B} \sum_{c=1}^{C} e_T(b, h, c)$$
Cập nhật 2 vector buffer `running_var_s` ($V_S \in \mathbb{R}^H$) và `running_var_t` ($V_T \in \mathbb{R}^H$, khởi tạo bằng $\mathbf{1}_H$) với hệ số làm mượt $m = 0.95$:
$$V_S(h) \leftarrow m \cdot V_S(h) + (1 - m) \cdot \frac{\hat{\sigma}_{S, \text{batch}}^2(h)}{\frac{1}{H}\sum_{k=1}^H \hat{\sigma}_{S, \text{batch}}^2(k) + \epsilon}$$
$$V_T(h) \leftarrow m \cdot V_T(h) + (1 - m) \cdot \frac{\hat{\sigma}_{T, \text{batch}}(h)}{\frac{1}{H}\sum_{k=1}^H \hat{\sigma}_{T, \text{batch}}(k) + \epsilon}$$

### Bước 3: Tính Trọng số Bước Thời gian từ Phương sai Thực nghiệm
- **Nhánh Seasonal ($\gamma_S > 0$ — Nghịch đảo phương sai nhiễu MLE):**
  Bước thời gian $h$ nào có phương sai sai số $V_S(h)$ càng lớn (nhiễu tần số cao không dự báo được) thì giảm trọng số đi:
  $$\tilde{w}_S(h) = \big(V_S(h) + \epsilon\big)^{-\gamma_S}, \quad w_S(h) = \frac{\tilde{w}_S(h)}{\frac{1}{H}\sum_{k=1}^H \tilde{w}_S(k)}$$
- **Nhánh Trend ($\gamma_T \ge 0$ — Giữ quỹ đạo xu hướng dài hạn):**
  Nhánh Trend đã qua bộ lọc thông thấp EMA; ở các bước cuối hay bị lệch quỹ đạo (co về trung bình), nên giữ phẳng ($\gamma_T = 0$) hoặc tăng nhẹ trọng số ở bước có độ lệch lớn ($\gamma_T > 0$):
  $$\tilde{w}_T(h) = \big(V_T(h) + \epsilon\big)^{+\gamma_T}, \quad w_T(h) = \frac{\tilde{w}_T(h)}{\frac{1}{H}\sum_{k=1}^H \tilde{w}_T(k)}$$

### Bước 4: Tổng hợp Loss theo chuẩn `DBLoss` ($\beta = 0.5$)
$$\mathcal{L}_{\text{Season}} = \frac{1}{B H C} \sum_{b,h,c} w_S(h) \cdot e_S(b, h, c), \quad \mathcal{L}_{\text{Trend}} = \frac{1}{B H C} \sum_{b,h,c} w_T(h) \cdot e_T(b, h, c)$$
$$\mathcal{L}_{\text{Trend}}^{\text{aligned}} = \mathcal{L}_{\text{Trend}} \cdot \text{detach}\!\left(\frac{\mathcal{L}_{\text{Season}}}{\mathcal{L}_{\text{Trend}} + \epsilon}\right)$$
$$\mathcal{L}_{\text{HD-DBLoss}} = \beta \cdot \mathcal{L}_{\text{Season}} + (1 - \beta) \cdot \mathcal{L}_{\text{Trend}}^{\text{aligned}}, \quad (\beta = 0.5)$$

---

## 3. Cấu Hình Thực Nghiệm Kiểm Chứng (`scripts/compare_losses.py`)

Hai cấu hình thực nghiệm được thiết lập hoàn toàn minh bạch, giữ nguyên $\alpha = 0.2, \beta = 0.5, m = 0.95$:
1. **`HD_DBLoss` ($\gamma_S = 0.5, \gamma_T = 0.0$):** Chỉ áp dụng trọng số nghịch đảo phương sai thực nghiệm trên nhánh Seasonal, giữ nguyên trọng số đều trên nhánh Trend.
2. **`HD_DBLoss_ST` ($\gamma_S = 0.5, \gamma_T = 0.25$):** Kết hợp giảm nhiễu nhánh Seasonal ($\gamma_S = 0.5$) và giữ quỹ đạo nhánh Trend ($\gamma_T = 0.25$).

### Lệnh chạy trên Google Colab:
```bash
git pull
python scripts/compare_losses.py --dataset ETTh1 --horizons 96 192 336 --losses DBLoss HD_DBLoss HD_DBLoss_ST
```
*(Bảng kết quả thực tế từ `result/compare_ETTh1.csv` sẽ được cập nhật vào Mục 4 ngay sau khi chạy xong trên GPU).*
