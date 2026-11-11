# Báo Cáo Nghiên Cứu & Thiết Kế Hàm Mất Mát TSG-DBLoss (`HD_DBLoss`)
**(Tri-Scale Spectral-Gated Decomposition Loss for Long-Term Time Series Forecasting)**

---

## 1. Bốn Nguyên Lý Bất Biến Cốt Lõi (4 Fundamental Invariances)

1. **Bất biến Đẳng Năng Lượng Theo Bước Thời Gian (*Iso-Energy Horizon Invariance* — Chống Bẫy Đánh Giá):**
   - Thước đo kiểm thử cuối cùng trên tập Test là MSE và MAE trung bình đều ($\frac{1}{H}\sum_{h=1}^H e_h^2$). Vì sai số tự nhiên tại các bước xa lớn gấp $3\text{–}5$ lần bước gần, mọi phương pháp giảm tổng trọng số tại bước xa ($w_{\text{total}}(h) < 1$) đều làm suy giảm lực kéo đạo hàm ở cuối cửa sổ dự báo, khiến Test MSE tăng vọt.
   - **Nguyên lý:** Tại mọi bước thời gian $h \in \{1, \dots, H\}$, tổng trọng số giữa hai nhánh phổ bảo toàn tuyệt đối bằng $1.0$:
     $$\forall h \in \{1, \dots, H\}, \qquad \beta(h) + \big(1 - \beta(h)\big) \equiv 1.0$$

2. **Bất biến Chống Vòng Lặp Phản Hồi Bỏ Rơi (*Zero Error-Driven Feedback Loop*):**
   - Loại bỏ hoàn toàn việc dùng sai số huấn luyện (`running_var_s`) để giảm trọng số. Mọi trọng số điều phối ($\beta(h)$, $w_c$) đều **độc lập 100% với dự đoán $\hat{Y}$ của mô hình**.

3. **Bất biến Định Hướng Đạo Hàm Chống Triệt Tiêu Adam (*Anti-Adam Cancellation*):**
   - Ma trận lọc nhân quả tam giác dưới $\mathbf{W}_{\alpha}$ liên kết đạo hàm của bước $k$ với toàn bộ các bước $h \ge k$. Khi tỷ lệ pha trộn phổ $\beta(h)$ biến thiên theo $h$, hướng của vector đạo hàm thay đổi (tỷ số giữa thành phần đạo hàm tần số cao và đạo hàm xu hướng tần số thấp trên cùng một tham số), đồng thời trọng số kênh $w_c$ tác động lên mạng chia sẻ tham số qua toàn bộ $C$ biến nên **không bị Adam chuẩn hóa mất hiệu lực**.

4. **Bất biến Dịch Chuyển & An Toàn Số Học `float32` (*Shift-Invariance & Numerical Safety*):**
   - Thay thế phép chia `cumsum` trong `EMA` gốc (vốn gây tràn số dưới `0.0 / 0.0 = NaN` khi $\alpha_{\text{fast}} \ge 0.6$ ở $H = 720$) bằng phép nhân ma trận nhân quả Toeplitz $\mathbf{W}_{\alpha}$ trực tiếp trên `float32` (`CausalEMA`).

---

## 2. Công Thức Toán Học Của Kiến Trúc **TSG-DBLoss**

### Bước 1: Phân Rã Tam Phân Nhân Quả ($X = T + S + R$)
Với chuỗi đầu vào $X \in \{\hat{Y}, Y\} \subset \mathbb{R}^{B \times H \times C}$ và ma trận nhân quả tam giác dưới $(\mathbf{W}_{\alpha})_{t,0} = (1-\alpha)^t$, $(\mathbf{W}_{\alpha})_{t,k} = \alpha(1-\alpha)^{t-k}$ ($1 \le k \le t$):
$$\begin{aligned}
T &= \text{CausalEMA}_{\alpha_{\text{slow}}}(X) &&\in \mathbb{R}^{B \times H \times C} \quad (\text{Xu hướng vĩ mô — } \alpha_{\text{slow}} = 0.2) \\
S &= \text{CausalEMA}_{\alpha_{\text{fast}}}(X - T) &&\in \mathbb{R}^{B \times H \times C} \quad (\text{Sóng chu kỳ điều hòa — } \alpha_{\text{fast}} = 0.6) \\
R &= (X - T) - S &&\in \mathbb{R}^{B \times H \times C} \quad (\text{Nhiễu ngẫu nhiên / gai đột biến cao tần})
\end{aligned}$$
*(Bảo toàn đẳng thức đại số $T + S + R \equiv X$; khi $\alpha_{\text{fast}} = 1.0 \implies R \equiv 0$, thu hồi chính xác `DECOMP` gốc).*

### Bước 2: Hàm Mất Mát Từng Dải Tần Khớp Độ Cong Bậc Hai
1. **Nhánh Sóng Chu Kỳ $S$ ($L_2$ Loss):**
   $$e_S(b, h, c) = (\hat{S}_{b,h,c} - S_{b,h,c})^2$$
2. **Nhánh Nhiễu Cao Tần $R$ (Curvature-Matched Huber Loss với ngưỡng $\delta = 1.0$):**
   $$e_R(b, h, c) = 2 \cdot \text{F.huber\_loss}(\hat{R}_{b,h,c}, R_{b,h,c}; \delta) = \begin{cases} (\hat{R}_{b,h,c} - R_{b,h,c})^2, & |\hat{R} - R| \le \delta \\ 2\delta|\hat{R} - R| - \delta^2, & |\hat{R} - R| > \delta \end{cases}$$
   $$e_{SR}(b, h, c) = e_S(b, h, c) + \lambda_R \, e_R(b, h, c) \quad (\lambda_R = 1.0)$$
3. **Nhánh Xu Hướng Vĩ Mô $T$ ($L_1$ Loss + Căn chỉnh Scale Toàn Cục $\kappa$):**
   $$e_T(b, h, c) = |\hat{T}_{b,h,c} - T_{b,h,c}|, \qquad \kappa = \text{detach}\!\left(\frac{\text{mean}(e_{SR})}{\text{mean}(e_T) + \epsilon}\right)$$

### Bước 3: Cổng Phổ Đẳng Năng Lượng Bảo Vệ Điểm Neo $\beta(h)$
Tại bước neo đầu tiên $h=1$ (nơi $S_1 = R_1 \equiv 0$), giữ $\beta(1) = \beta$; trên miền dao động $h \in \{2, \dots, H\}$, điều biến đối xứng:
$$\beta(h) = \begin{cases} \beta, & h = 1 \\ \beta + \Delta\beta_{\text{eff}} \cdot \left(1 - 2\frac{h - 2}{H - 2}\right), & h \in \{2, \dots, H\} \end{cases}$$
với $\Delta\beta_{\text{eff}} = \max\!\big(0, \min(\Delta\beta, \beta - 0.02, 0.98 - \beta)\big)$ ($\Delta\beta = 0.15$).

### Bước 4: Trọng Số Kênh Theo SNR Bất Biến Dịch Chuyển ($w_c$)
Khử trung bình cục bộ cửa sổ $\widetilde{T}_{b,h,c} = T_{b,h,c} - \frac{1}{H}\sum_{k=1}^H T_{b,k,c}$ trên nhãn thực tế $Y$:
$$\text{SNR}_c = \frac{\frac{1}{BH}\sum_{b,h} \left(\widetilde{T}_{b,h,c}^2 + S_{b,h,c}^2\right)}{\frac{1}{BH}\sum_{b,h} R_{b,h,c}^2 + \epsilon}, \qquad w_c = \frac{(\text{SNR}_c + \epsilon)^\eta}{\frac{1}{C}\sum_{j=1}^C (\text{SNR}_j + \epsilon)^\eta} \quad (\eta = 0.05)$$

### Bước 5: Biểu Thức Tổng Hợp Cuối Cùng
$$\mathcal{L}_{\text{TSG-DBLoss}}(\hat{Y}, Y) = \frac{1}{B H C} \sum_{b=1}^{B} \sum_{h=1}^{H} \sum_{c=1}^{C} w_c \cdot \Big[ \beta(h) \, e_{SR}(b, h, c) + \big(1 - \beta(h)\big) \, \kappa \, e_T(b, h, c) \Big]$$

---

## 3. Lệnh Chạy Thực Nghiệm Ablation Trên Google Colab

```bash
git pull
# Cách 1: Chạy với batch_size chuẩn bài báo (tái sử dụng cache DBLoss)
python scripts/compare_losses.py --dataset ETTh1 --horizons 96 192 336

# Cách 2: Chạy nhanh với batch_size=128 (so sánh công bằng ở cùng batch_size=128)
python scripts/compare_losses.py --dataset ETTh1 --horizons 96 192 336 --batch-size 128
```
