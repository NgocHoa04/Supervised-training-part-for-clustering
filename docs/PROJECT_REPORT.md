# Project Report — Supervised Training for Clustering

---

## 1. Multi-Class Metrics & Evaluation

### 1.1 Tại sao đây là bài toán phân loại đa lớp?

Dataset có 4 cluster (0, 1, 2, 3) được gán từ bước phân cụm trước. Nhiệm vụ của Supervised Learning là học cách phân loại một customer vào đúng cluster dựa trên các features quan sát được — đây là multi-class classification với 4 lớp.

---

### 1.2 Các metric quan trọng

#### ACCURACY

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Ý nghĩa:** Tỷ lệ dự đoán đúng trên toàn bộ tập dữ liệu.

**Dùng khi:** Các class có kích thước cân bằng và mọi lỗi đều quan trọng như nhau.

**Hạn chế:** Bị méo khi dữ liệu mất cân bằng — model có thể đạt accuracy cao chỉ bằng cách luôn dự đoán class đa số.

---

#### PRECISION

$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$$

**Ý nghĩa:** Trong tất cả những gì model dự đoán là class $i$, bao nhiêu phần trăm thực sự là class $i$?

**Dùng khi:** False Positive tốn kém (ví dụ: spam detection — không muốn chặn nhầm email quan trọng).

---

#### RECALL (Sensitivity)

$$\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$$

**Ý nghĩa:** Trong tất cả các mẫu thực sự thuộc class $i$, model tìm được bao nhiêu phần trăm?

**Dùng khi:** False Negative tốn kém (ví dụ: phát hiện gian lận — không muốn bỏ sót case gian lận).

---

#### F1-SCORE (Harmonic Mean)

$$F1_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

**Ý nghĩa:** Cân bằng giữa Precision và Recall. Harmonic mean bị kéo mạnh về phía giá trị thấp hơn — nếu một trong hai rất thấp, F1 sẽ thấp theo.

**Macro F1:** Trung bình cộng đơn giản F1 của từng class — coi tất cả class quan trọng như nhau.

**Weighted F1:** Trung bình có trọng số theo kích thước class — phản ánh hiệu suất tổng thể sát thực tế hơn.

**Dùng khi:** Dữ liệu mất cân bằng, cần cân bằng cả hai loại lỗi. **Metric được khuyến nghị nhất cho bài toán này.**

---

#### BALANCED ACCURACY

$$\text{Balanced Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \text{Recall}_i$$

**Ý nghĩa:** Trung bình Recall trên tất cả các class — đảm bảo hiệu suất đều nhau giữa các class dù kích thước chênh lệch.

**Dùng khi:** Dữ liệu mất cân bằng, muốn đánh giá khả năng phân loại đồng đều trên mọi class.

---

#### COHEN'S KAPPA

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Trong đó $p_o$ là accuracy quan sát được, $p_e$ là accuracy kỳ vọng nếu model đoán ngẫu nhiên theo phân phối class.

**Ý nghĩa:** Đo lường mức độ model tốt hơn việc đoán ngẫu nhiên — loại bỏ ảnh hưởng của cơ cấu class.

**Thang đánh giá:** $\kappa > 0.9$ xuất sắc | $0.8$–$0.9$ rất tốt | $0.6$–$0.8$ tốt | $0.4$–$0.6$ trung bình | $< 0.4$ kém.

**Dùng khi:** Dữ liệu mất cân bằng, muốn so sánh công bằng giữa các model.

---

#### MATTHEWS CORRELATION COEFFICIENT (MCC)

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Ý nghĩa:** Hệ số tương quan giữa nhãn thực và nhãn dự đoán — tính đến cả 4 thành phần của confusion matrix. Phạm vi [-1, 1]: 1 là hoàn hảo, 0 là ngẫu nhiên, -1 là hoàn toàn sai.

**Dùng khi:** Dữ liệu mất cân bằng nghiêm trọng, muốn metric toàn diện và không bị thiên vị nhất.

---

#### CONFUSION MATRIX

Bảng $n \times n$ (với $n$ = số class) thể hiện:
- **Đường chéo:** số dự đoán đúng của từng class
- **Ngoài đường chéo:** số dự đoán sai — hàng là nhãn thực, cột là nhãn dự đoán

**Dùng khi:** Cần phân tích chi tiết loại lỗi — model nhầm cluster nào với cluster nào nhiều nhất.

---

#### JACCARD SCORE

$$\text{Jaccard}_i = \frac{TP_i}{TP_i + FP_i + FN_i}$$

**Ý nghĩa:** Tỷ lệ giao/hợp giữa tập dự đoán và tập thực tế — nhạy cảm với cả FP lẫn FN.

**Dùng khi:** Muốn đánh giá nghiêm ngặt (bất kỳ lỗi nào đều bị phạt), thường dùng trong object detection.

---

### 1.3 Khuyến nghị chọn metric cho bài toán này

| Tình huống | Metric nên dùng |
|---|---|
| Đánh giá tổng quan | Accuracy, F1-Weighted |
| Class mất cân bằng | Balanced Accuracy, F1-Macro, Cohen's Kappa, MCC |
| Phân tích lỗi chi tiết | Confusion Matrix + Classification Report |
| So sánh model | F1-Weighted (primary), MCC (secondary) |

---

## 2. Hyperparameter Tuning & Overfitting Analysis

### 2.1 Các model đã train

Notebook `02_complete_analysis.ipynb` train và tune 6 model: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree. Notebook `03_logistics_regression.ipynb` deep-dive riêng cho Logistic Regression với Optuna.

**Phương pháp tuning:**
- `02_complete_analysis.ipynb`: Grid Search CV (5-fold StratifiedKFold) cho tất cả model
- `03_logistics_regression.ipynb`: Bayesian Optimization bằng **Optuna** (TPE Sampler, 100 trials) cho Logistic Regression

### 2.2 Không gian tham số Logistic Regression (Optuna)

| Tham số | Phạm vi |
|---|---|
| `C` (regularization) | $10^{-4}$ đến $10^2$ (log scale) |
| `penalty` | l1, l2, elasticnet, None |
| `solver` | lbfgs, saga, newton-cg, sag (tùy penalty) |
| `class_weight` | None, balanced |
| `l1_ratio` | 0.0–1.0 (chỉ khi penalty=elasticnet) |

Cross-validation: StratifiedKFold 5-fold, tối ưu F1-Macro.

### 2.3 Kết quả model Logistic Regression

**Baseline (default params):**

| Set | Accuracy | F1-Macro | F1-Weighted | Balanced Accuracy |
|---|---|---|---|---|
| Train | — | — | — | — |
| Test | ~90% | ~90% | ~90% | ~90% |

- Cluster 0, 1, 3 phân loại xuất sắc
- Cluster 2: đúng 185/190 (97.4%) — cluster nhỏ, khó phân biệt nhất
- **Lỗi chính:** 136 mẫu cluster 0 bị nhầm thành cluster 3 (hai cluster này có đặc trưng tương đồng)

**Optuna Tuned:**

| Metric | Baseline | Optuna Tuned | Thay đổi |
|---|---|---|---|
| F1-Macro | ~0.90 | **0.9175** | ↑ |
| Cluster 2 accuracy | 185/190 | **190/190** | ↑ (100%) |
| Cluster 0 vs 3 confusion | 136 nhầm | 162 nhầm | ↑ (trade-off) |

**Nhận xét về Optuna Optimization History:**
- 100 trials đạt F1-Macro ổn định quanh 0.92
- 3 trials thất bại (F1 = 0.15–0.57) do incompatible solver-penalty pairs
- Đường red line (best score) phẳng sớm → Optuna tìm được tham số tốt từ sớm, có thể dừng ở ~50 trials
- **Tham số C (regularization strength) quan trọng nhất** — ảnh hưởng lớn nhất đến hiệu suất; các tham số khác (solver, penalty type) ít quan trọng hơn

**Best F1-Score đạt được: 91.75%** (Optuna Tuned, toàn bộ features)

### 2.4 Kiểm tra overfitting

$$\text{Overfitting Gap} = \text{Train Score} - \text{Test Score}$$

**Ngưỡng đánh giá:**

| Gap | Đánh giá |
|---|---|
| < 2% | ✅ Tốt — không overfitting |
| 2%–10% | ⚠️ Overfitting nhẹ |
| > 10% | ❌ Overfitting nghiêm trọng — cần regularization |

**Learning curve analysis (notebook 03):** Train score và Validation score hội tụ gần nhau khi tăng dần kích thước train set → model không bị overfit. Khoảng cách gap nhỏ và ổn định → **mô hình có generalization tốt.**

---

## 3. Feature Importance Analysis

### 3.1 Phương pháp tính feature importance

**Tree-based models (Random Forest, Gradient Boosting, Decision Tree):**
- Gini importance: tổng lượng giảm impurity tại mỗi split có dùng feature đó, được chuẩn hóa theo toàn cây

**Linear models (Logistic Regression):**
- Trị tuyệt đối của hệ số: $|coef_j|$ — sau khi chuẩn hóa feature, hệ số lớn hơn = đóng góp lớn hơn vào decision boundary

**ElasticNet-based feature selection (notebook 03):**
- Dùng cả L1 (sparsity) và L2 (stability): $\text{penalty} = \alpha[\rho ||w||_1 + (1-\rho)||w||_2^2]$
- Tìm $l1\_ratio$ tối ưu qua cross-validation trên 5 giá trị {0.1, 0.3, 0.5, 0.7, 0.9}
- Features với coefficient về 0 → bị loại bỏ tự động

**Consensus approach (notebook 02):**
1. Chuẩn hóa importance của từng model (sum-to-1)
2. Tính trung bình và độ lệch chuẩn qua các model
3. Độ lệch chuẩn thấp → các model đồng thuận về tầm quan trọng của feature

### 3.2 Kết quả so sánh full features vs top features

**Model tốt nhất: Optuna Tuned Logistic Regression (toàn bộ features)**

| Model | Test F1-Macro | Test Accuracy | Số features |
|---|---|---|---|
| Baseline (all features) | ~0.90 | ~0.90 | Toàn bộ |
| **Optuna Tuned (all features)** | **0.9175** | **~0.92** | Toàn bộ |
| Optuna + Feature Selection (top N) | Thấp hơn | Thấp hơn | Top N |

**Kết luận quan trọng:** Giữ toàn bộ features + regularization mạnh **thắng** feature selection. Lý do: regularization "làm im" các feature yếu thay vì loại bỏ hoàn toàn — những feature "yếu" đơn lẻ có thể tạo pattern mạnh khi kết hợp với nhau. Model full features nắm bắt được các tương tác ẩn mà model rút gọn bỏ mất.

### 3.3 Top features quan trọng nhất

Dựa trên consensus từ nhiều model và ElasticNet analysis:

| Hạng | Feature | Ý nghĩa |
|---|---|---|
| 1 | `Months_Inactive_12_mon` | Số tháng không hoạt động trong 12 tháng |
| 2 | `Total_Trans_Amt` | Tổng số tiền giao dịch |
| 3 | `Total_Trans_Ct` | Tổng số lần giao dịch |
| 4 | `Total_Revolving_Bal` | Số dư revolving |
| 5 | `Total_Ct_Chng_Q4_Q1` | Thay đổi số lần giao dịch Q4/Q1 |

---

## 4. Cluster Analysis & Business Insights

### 4.1 Profile từng cluster

*(Dựa trên 2 markdown cells cuối notebook `02_complete_analysis.ipynb` và `03_logistics_regression.ipynb`)*

---

**Cluster 0 — Dormant Customers**

*Driver chính:* Số tháng không hoạt động rất cao (+3.5)

*Các yếu tố phụ:* Tổng chi tiêu (+0.15), số giao dịch (+0.1) — nhỏ nhưng dương; credit limit thấp (-0.07), tuổi trẻ hơn (-0.06)

*Mô tả:* Khách hàng có thời gian không hoạt động dài, credit limit thấp, thỉnh thoảng vẫn có giao dịch nhỏ nhưng không đáng kể. Nhóm khách hàng đang dần "ngủ đông" với ngân hàng.

*Business action:* **Re-engagement campaigns** — ưu đãi kích hoạt lại (cashback, lãi suất 0%), cá nhân hóa theo profile tuổi trẻ.

---

**Cluster 1 — Long-term, Low-Activity Customers**

*Driver chính:* Thời gian gắn bó lâu (+0.2), số sản phẩm ngân hàng nhiều (+0.2), thẻ cao cấp (+0.2), số tháng không hoạt động vừa phải (+0.5)

*Yếu tố âm mạnh:* Tổng chi tiêu (-20.0), số giao dịch (-17.5) — **rất mạnh**

*Mô tả:* Khách hàng lâu năm, có nhiều sản phẩm (tài khoản, thẻ, tiết kiệm) nhưng rất ít sử dụng. Kiểu khách hàng "relationship banking" — duy trì tài khoản nhưng không chủ động giao dịch.

*Business action:* **Activation programs** — cross-sell dựa trên portfolio sản phẩm hiện có, khuyến khích dùng thẻ qua rewards.

---

**Cluster 2 — High-Value Active Customers**

*Driver chính:* Số lần giao dịch (+7.5), tổng chi tiêu (+6.5) — **rất mạnh**

*Yếu tố phụ dương:* Credit limit cao (+0.1)

*Yếu tố âm:* Thu nhập (-0.3), xu hướng giao dịch giảm (-0.25), trình độ học vấn (-0.25), số sản phẩm ngân hàng (-0.2)

*Mô tả:* Khách hàng giao dịch nhiều nhất và chi tiêu cao nhất. Đây là nhóm **valuable nhất về transaction**. Đặc biệt: họ có ít sản phẩm ngân hàng hơn nhưng dùng thẻ rất tích cực.

*Business action:* **Premium services & loyalty programs** — cashback tier cao, airport lounge, concierge service. Cơ hội upsell sản phẩm vì họ có engagement cao nhưng số sản phẩm thấp.

---

**Cluster 3 — Growth-Oriented Customers**

*Driver chính:* Số tháng không hoạt động rất thấp (-6.0 — tức là "hiếm khi ngừng hoạt động")

*Xu hướng tăng trưởng:* Tần suất giao dịch tăng (+0.12), chi tiêu tăng (+0.06) so với đầu năm

*Yếu tố nhân khẩu học nhỏ:* Ly hôn (+0.1), trình độ học vấn cao (+0.04)

*Mô tả:* Khách hàng đang trong giai đoạn **tăng trưởng engagement**. Họ gần như không bao giờ ngừng hoạt động và đang dùng thẻ ngày càng nhiều hơn. Có thể liên quan đến life events (ly hôn → thay đổi tài chính độc lập).

*Business action:* **Personalized growth incentives** — nurture relationship, offer sản phẩm phù hợp với life stage, loyalty program để duy trì đà tăng trưởng.

---

### 4.2 Kết luận: Features có nói lên điều gì về các cluster không?

**Có — và rất rõ ràng. Đây là "câu chuyện trung tâm" của dataset:**

**Transaction behavior là yếu tố phân biệt chính.** Các features liên quan đến giao dịch (`Total_Trans_Ct`, `Total_Trans_Amt`, `Total_Ct_Chng_Q4_Q1`) và inactivity (`Months_Inactive_12_mon`) là những features mạnh nhất, thống nhất trên mọi model.

**Ranh giới rõ nhất: Inactivity — bức tường phân chia:**

| Nhóm engaged | Nhóm disengaged |
|---|---|
| Cluster 2 & 3 | Cluster 0 & 1 |
| Giao dịch thường xuyên | Ít hoặc gần như không giao dịch |
| Inactivity thấp | Inactivity cao |

**Tension giữa "sản phẩm" và "sử dụng":**
- Cluster 1: nhiều sản phẩm, ít dùng → "relationship without engagement"
- Cluster 2: ít sản phẩm, dùng nhiều → "usage without breadth"
- Đây là opportunity để cross-sell cho Cluster 2 và activation cho Cluster 1

**Demographics ít quan trọng hơn behavior.** Thu nhập, học vấn, tuổi chỉ là yếu tố phụ — **hành vi khách hàng quan trọng hơn đặc điểm nhân khẩu học.** Điều này hoàn toàn hợp lý về mặt business: một người thu nhập cao nhưng không dùng thẻ vẫn có giá trị thấp hơn người thu nhập trung bình nhưng giao dịch thường xuyên.

**Compatibility với business sense:** ✅ **Hoàn toàn hợp lý.**
- Ngân hàng thực tế phân nhóm khách hàng theo mức độ engagement và transactional value
- Nhóm dormant (Cluster 0 & 1) là rủi ro churn cao nhất
- Nhóm High-Value (Cluster 2) cần retention strategy đặc biệt
- Nhóm Growth (Cluster 3) là cơ hội tăng revenue lớn nhất nếu được nurture đúng cách

---

*Report được tổng hợp từ: `02_complete_analysis.ipynb` (model comparison, feature importance, cluster analysis) và `03_logistics_regression.ipynb` (Optuna tuning, overfitting check, ElasticNet feature selection).*
