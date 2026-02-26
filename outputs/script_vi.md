# Hướng dẫn thuyết trình (15 phút)

**Đề tài:** A Vision-First Multi-Modal Approach for Hand Gesture Recognition

> Tài liệu này giải thích từng slide, cung cấp lời thoại mẫu, và chuẩn bị cho phần Q&A.

---

## Tổng quan dự án (đọc trước khi trình bày)

### Bài toán

Xây dựng game xếp hình (puzzle) trên trình duyệt, điều khiển hoàn toàn bằng cử chỉ tay qua webcam, không cần server, không cần cài đặt. Người dùng mở trang web, bật camera, và chơi ngay.

### Pipeline tổng quát

```
Webcam → MediaPipe (phát hiện tay) → MLP (phân loại cử chỉ) → Mô hình toán (lọc jitter) → Game
```

- **MediaPipe** (của Google, có sẵn): phát hiện tay, trả về 21 điểm 3D trên bàn tay (landmark).
- **MLP** (tự train): nhận 60 số (tọa độ 20 landmark × 3 chiều, bỏ cổ tay) → phân loại thành 5 cử chỉ.
- **Mô hình toán** (tự xây dựng, adapted từ Huda et al. 2025): lọc nhiễu tay, biến phân loại từng frame thành hành động game mượt mà.
- **Game** (tự xây dựng): puzzle HTML5 Canvas chạy 100% trên trình duyệt.

### Ba câu hỏi nghiên cứu

| # | Câu hỏi | Phát hiện |
|---|---------|-----------|
| Q1 | Fusion skeleton + appearance có cải thiện accuracy không? | Ceiling effect — cả hai modality đã bão hòa (~100%) nên fusion không cải thiện thêm. Đây là boundary condition có giá trị khoa học. |
| Q2 | Model tối thiểu nào đủ cho real-time browser deployment? | MLP 14.7K params (227 KB ONNX) trên skeleton landmarks — nhẹ nhất mà vẫn đạt 99.8%. |
| Q3 | Cần gì để biến classification thành interactive control? | Mô hình toán học: grid-based swipe (Huda et al. 2025) + debounce 3-frame + adaptive cursor smoothing. |

### Ba đóng góp

1. **So sánh multi-modal**: 5 phương pháp chính (+ heuristic baseline), phát hiện ceiling effect (kết quả âm có giá trị).
2. **Mô hình tương tác toán học**: adapted từ Huda et al. 2025 (điều khiển xe lăn → game xếp hình).
3. **Triển khai trên trình duyệt**: 227 KB, 30+ FPS, không server.

---

## Slide-by-slide

---

### Slide 1: Title (30 giây)

**Nội dung:** Tên đề tài, tên nhóm, môn học.

**Nói:**
> "Chào thầy/cô và các bạn. Nhóm chúng em trình bày đề tài 'A Vision-First Multi-Modal Approach for Hand Gesture Recognition'. Chúng em nghiên cứu cách nhận dạng cử chỉ tay bằng nhiều phương pháp thị giác máy tính, và ứng dụng vào điều khiển game xếp hình trên trình duyệt."

---

### Slide 2: Research Questions (1 phút)

**Nội dung:** Pipeline tổng quát (Webcam → MediaPipe → Classify → Control → Game) và 3 câu hỏi nghiên cứu.

**Giải thích cho bản thân:**

- Pipeline gồm 5 bước. MediaPipe là bước "cảm biến" — nó phát hiện tay và trả về tọa độ landmark. Phần còn lại là công việc của chúng ta.
- **Q1 (Multi-Modal):** "Multi-modal" nghĩa là dùng nhiều loại dữ liệu đầu vào khác nhau. Ở đây chúng ta có 2 modality: skeleton (hình dạng bộ xương tay, 60 con số) và appearance (ảnh RGB 224×224 pixel của bàn tay). Câu hỏi: kết hợp 2 cái này có tốt hơn dùng 1 cái không?
- **Q2 (Model Size):** Vì chạy trên trình duyệt, model càng nhỏ càng tốt. Model nào nhỏ nhất mà vẫn đủ chính xác?
- **Q3 (Interaction):** Phân loại đúng 99.8% nghe hay, nhưng nếu map thẳng vào game thì tay run → cursor giật → tile nhảy lung tung. Làm sao để lọc nhiễu?

**Nói:**
> "Chúng em đặt ra ba câu hỏi nghiên cứu. Thứ nhất: liệu việc kết hợp đặc trưng skeleton và appearance có cải thiện kết quả không? Thứ hai: model nhỏ nhất đủ dùng cho browser game là gì? Thứ ba: làm sao biến kết quả phân loại từng frame thành điều khiển game mượt mà, không bị giật?"

---

### Slide 3: Dataset (1.5 phút)

**Nội dung:** HaGRID v2, 5 classes, hình skeleton per class, biểu đồ phân bố.

**Giải thích cho bản thân:**

- **HaGRID v2** là bộ dữ liệu lớn nhất cho nhận dạng cử chỉ tay, do Sber (Nga) công bố. Bản đầy đủ có 554K ảnh, 18 class, 37K người.
- Chúng ta chọn **subset 5 class** phù hợp với game:
  - **Pinch** (nhúm tay): dùng để "nhặt" tile
  - **Fist** (nắm tay): dùng để reset puzzle
  - **Open Hand** (xòe tay): dùng để "thả" tile
  - **Frame** (khung hình 2 tay): dùng để chụp ảnh cho puzzle
  - **None** (không cử chỉ): không làm gì
- **23,850 mẫu** từ **13,250 người** khác nhau. Mỗi người chỉ xuất hiện 1-2 lần → không bị overfit theo danh tính.
- **Mất cân bằng 3:1:** Frame và None có 7,950 mẫu mỗi class (vì 1 ảnh có nhiều tay → nhiều crop), 3 class còn lại chỉ có 2,650. Chúng ta xử lý bằng stratified split và báo cáo per-class F1.

**Nói:**
> "Chúng em sử dụng subset 5 class từ HaGRID v2 — bộ dataset lớn nhất cho hand gesture recognition — với gần 24 nghìn mẫu từ hơn 13 nghìn người khác nhau. Năm cử chỉ được chọn tương ứng với các hành động trong game: pinch để nhặt tile, open hand để thả, fist để reset, frame để chụp ảnh, và none là trạng thái nghỉ. Dataset có tỷ lệ mất cân bằng 3:1 giữa Frame/None và các class còn lại."

---

### Slide 4: Two Feature Streams (1.5 phút)

**Nội dung:** Sơ đồ dual-stream (Camera → Skeleton/Appearance → MLP/CNN → Fusion), t-SNE plot, và phương pháp đánh giá.

**Giải thích cho bản thân:**

- **"Two modalities from one camera"** = từ cùng 1 frame webcam, chúng ta trích xuất 2 loại đặc trưng hoàn toàn khác nhau:
  - **Skeleton (hình học):** MediaPipe trả về 21 điểm 3D → chuẩn hóa → 60 con số. Đây là "bộ xương" của bàn tay. Nó nắm bắt hình dạng (ngón cong/thẳng, khoảng cách giữa các ngón).
  - **Appearance (hình ảnh):** Cắt vùng bàn tay từ ảnh gốc, resize về 224×224 pixel. Đây là "bề ngoài" — màu da, kết cấu, hình dáng tổng thể.
- **t-SNE** là phương pháp giảm chiều để trực quan hóa. Plot cho thấy 5 class đã tách biệt rõ ràng trong không gian 60D → dự báo rằng skeleton alone sẽ đủ.
- **Person-aware Group 5-Fold CV:** Đây là phương pháp đánh giá nghiêm ngặt nhất. Dùng subject ID làm group label → đảm bảo không ai xuất hiện ở cả train lẫn test. 13,250 người × 5 folds = mỗi người đều được test như "người lạ". Nếu dùng random split (không group), accuracy có thể bị thổi phồng vì model nhớ mặt người thay vì học cử chỉ.

**Nói:**
> "Từ cùng một camera, chúng em trích xuất hai luồng đặc trưng. Skeleton: MediaPipe cho 21 điểm 3D, chuẩn hóa thành vector 60 chiều. Appearance: crop bàn tay thành ảnh 224×224 RGB. Biểu đồ t-SNE cho thấy chỉ riêng skeleton 60 chiều đã phân tách rõ 5 class — gợi ý rằng skeleton alone có thể đủ. Đánh giá bằng Person-aware Group 5-Fold CV: không ai xuất hiện ở cả train lẫn test, đảm bảo kết quả phản ánh khả năng tổng quát hóa thực sự."

---

### Slide 5: Five Methods + Results (1.5 phút)

**Nội dung:** Bảng 5 phương pháp (RF → MLP → CNN → YOLO → Fusion), biểu đồ accuracy và per-class F1.

**Giải thích cho bản thân:**

5 phương pháp trên chart/table (heuristic bỏ khỏi bảng vì quá thấp, chỉ đề cập bằng lời):

1. **Random Forest (94.1%):** 100 cây quyết định trên 24 đặc trưng khoảng cách (fingertip-to-fingertip, etc.). Tốt nhưng nhầm lẫn Frame↔None vì 24 features không đủ phân biệt. *Bằng chứng rằng feature engineering có giới hạn.*
2. **MLP (99.8%, 227 KB):** Mạng 3 lớp (60→128→64→5) trực tiếp trên 60D vector. Gần hoàn hảo với chỉ 14.7K tham số. *Đây là model được deploy.*
3. **CNN/MobileNetV3-Small (100.0%):** Fine-tune trên ảnh crop 224×224. Hoàn hảo nhưng nặng hơn (1.5M params). *Bằng chứng rằng appearance cũng saturate.*
4. **YOLOv8n (99.5% mAP@50):** Phát hiện + phân loại đồng thời. Metric khác (mAP thay vì accuracy) vì đây là bài toán detection. 3M params, 6.2 MB. *Quá nặng cho browser.*
5. **Fusion (100.0%):** Trung bình có trọng số softmax outputs của MLP và CNN (α=0.7 cho MLP). Bằng CNN, không hơn. *Bằng chứng cho ceiling effect.*

> *(Ngoài lề — mention bằng lời nếu được hỏi:)* Heuristic (luật tay, 1.0%) chỉ đạt ~1% → đã bỏ khỏi bảng vì outlier quá lớn. Giải thích: ~20 luật if-else thiết kế cho 3D nhưng MediaPipe trả 2D chiếu → mất depth info.

**Câu chuyện mà progression kể:** Từ ML cổ điển (feature engineering, 94%) → neural net (saturate, 99.8-100%) → fusion (không thêm gì). Kết luận: bài toán quá dễ cho model hiện đại.

**Nói:**
> "Chúng em đánh giá 5 phương pháp chính theo độ phức tạp tăng dần. Random Forest đạt 94% nhưng nhầm lẫn Frame và None. MLP trên full 60 chiều đạt 99.8% chỉ với 227 kilobyte. CNN trên ảnh RGB đạt 100%. YOLO detection baseline đạt 99.5% mAP. Và fusion — kết hợp MLP + CNN — cũng đạt 100%, bằng CNN, không hơn. Progression này cho thấy: neural net saturate, fusion không thêm gì."

---

### Slide 6: Where Do Errors Happen? (1 phút)

**Nội dung:** Confusion matrices của Random Forest (94%) và MLP (99.8%).

**Giải thích cho bản thân:**

- **Confusion matrix** là bảng so sánh dự đoán vs thực tế. Đường chéo = đúng, ngoài chéo = sai.
- **RF confusion:** Frame và None bị nhầm lẫn nhiều. Lý do: 24 đặc trưng khoảng cách không đủ phân biệt "không cử chỉ" (None) với "hai tay tạo khung" (Frame) vì cả hai đều có ngón tay mở.
- **MLP confusion:** Gần như hoàn hảo. Full 60D vector (tất cả tọa độ x,y,z) chứa đủ thông tin để phân biệt. Lỗi còn lại tập trung ở class None — vì None là "heterogeneous negative" (bất kỳ tư thế nào không phải 4 cử chỉ kia).

**Nói:**
> "Confusion matrix cho thấy Random Forest nhầm nhiều giữa Frame và None — 24 đặc trưng khoảng cách không đủ phân biệt. MLP trên full 60 chiều giải quyết hoàn toàn, chỉ còn lỗi nhỏ trên class None — vì None là tập hợp mọi tư thế không thuộc 4 cử chỉ kia."

---

### Slide 7: YOLOv8n (1 phút)

**Nội dung:** Kết quả YOLO, so sánh trade-off với MLP.

**Giải thích cho bản thân:**

- **YOLO** là mô hình end-to-end: nhận ảnh gốc → vừa tìm tay (bounding box) vừa phân loại (class label). Không cần MediaPipe.
- **mAP@50 = 99.5%:** Trong 100 tay cần phát hiện, 99.5 tay được phát hiện đúng với IoU > 50%.
- **mAP@50:95 = 90.7%:** Metric khắt khe hơn (yêu cầu bbox chính xác hơn).
- **Tại sao không dùng YOLO?** MediaPipe đã cung cấp hand detection miễn phí. YOLO nặng gấp 27 lần MLP (6.2 MB vs 227 KB). Thêm localization overhead mà ta không cần.
- **Tại sao vẫn train YOLO?** Để có baseline end-to-end so sánh, và làm fallback nếu MediaPipe fail.

**Nói:**
> "YOLO là baseline end-to-end — phát hiện và phân loại đồng thời. Đạt 99.5% mAP, nhưng nặng gấp 27 lần MLP. Vì MediaPipe đã cung cấp hand detection, YOLO thêm overhead localization không cần thiết. Kết luận: MLP trên MediaPipe landmarks là trade-off tốt nhất cho browser deployment."

---

### Slide 8: Ceiling Effect (1.5 phút)

**Nội dung:** Core finding — cả hai modality đều saturate, fusion không thêm gì.

**Giải thích cho bản thân:**

- **Ceiling effect** = "hiệu ứng trần": khi accuracy đã chạm trần (~100%), không phương pháp nào cải thiện thêm được. Giống như thi 10 điểm rồi — không có cách nào lên 11.
- **Tại sao xảy ra?**
  1. 5 cử chỉ khác nhau hoàn toàn về mặt hình học: nắm tay ≠ xòe tay ≠ nhúm tay. Không ai nhầm.
  2. Person-aware CV đảm bảo không có data leakage → kết quả đáng tin.
  3. 13,250 người → model tổng quát hóa tốt.
- **Đây là negative result nhưng có giá trị:** Nó trả lời câu hỏi "khi nào cần multi-modal?" bằng data: khi task đơn giản (ít class, class khác biệt rõ), 1 modality đủ. Multi-modal chỉ có ý nghĩa khi task khó hơn (ví dụ: full HaGRID 18 class, SOTA chỉ 95.5%).
- **Implication thực tế:** Deploy MLP 227 KB. Không cần CNN, không cần fusion.

**Nói:**
> "Đây là phát hiện quan trọng nhất: cả skeleton lẫn appearance đều độc lập đạt gần 100% accuracy. Fusion không thêm bất kỳ cải thiện nào — một ceiling effect. Điều này xảy ra vì 5 cử chỉ của chúng ta khác nhau hoàn toàn về mặt hình học. Đây là kết quả âm có giá trị — nó xác định boundary condition: với gesture vocabulary đơn giản, single-modality MLP là đủ. Multi-modal fusion chỉ có ý nghĩa khi task khó hơn, ví dụ full HaGRID 18 class."

---

### Slide 9: Classification ≠ Control (1.5 phút)

**Nội dung:** So sánh trực quan: không có interaction model (cursor giật) vs có interaction model (smooth).

**Giải thích cho bản thân:**

Đây là slide "twist" — chuyển từ "phân loại xong rồi" sang "vấn đề thực sự mới bắt đầu":

- **Vấn đề 1: Jitter.** Tay người run tự nhiên 5-15 pixel/frame. Nếu map thẳng tọa độ tay → cursor, cursor sẽ nhảy liên tục → vô tình trigger swap tile.
- **Vấn đề 2: Transient errors.** 99.8% accuracy nghe cao, nhưng ở 30 FPS = 30 dự đoán/giây. 0.2% lỗi = khoảng 1 frame sai mỗi 17 giây. Nếu frame sai đó xảy ra giữa lúc đang cầm tile → thả tile giữa chừng.
- **Giải pháp:** Mô hình toán học (slide sau) lọc cả hai vấn đề.
- **Nguồn cảm hứng:** Huda et al. 2025 gặp đúng vấn đề này khi điều khiển xe lăn bằng cử chỉ tay. Họ đạt 98.14% accuracy nhưng vẫn cần threshold model để xe chạy ổn.

**Nói:**
> "99.8% accuracy nghe tuyệt vời, nhưng classification không bằng control. Tay người run tự nhiên 5-15 pixel mỗi frame — nếu map thẳng vào game, cursor giật và tile nhảy lung tung. 0.2% lỗi ở 30 FPS nghĩa là khoảng 1 frame sai mỗi 17 giây — đủ để thả tile giữa chừng. Huda et al. 2025 gặp đúng vấn đề này khi điều khiển xe lăn — họ giải bằng mô hình toán. Chúng em adapt cách tiếp cận đó."

---

### Slide 10: Mathematical Interaction Model (2 phút) — SLIDE QUAN TRỌNG NHẤT

**Nội dung:** Sơ đồ tolerance zone, bảng điều kiện swipe, tham số.

**Giải thích cho bản thân:**

Đây là phần adapted từ Huda et al. 2025. Ý tưởng cốt lõi:

**Thay vì free-drag (kéo tự do), dùng threshold-gated swap (vuốt có ngưỡng):**

1. Khi người dùng pinch (nhúm tay) trên 1 tile → ghi nhận điểm bắt đầu (x_init, y_init).
2. Khi tay di chuyển, tính displacement: Δx = x_hiện_tại - x_bắt_đầu, Δy = y_hiện_tại - y_bắt_đầu.
3. **Chỉ trigger swap khi đủ hai điều kiện:**
   - **Movement threshold (mov):** Phải di chuyển đủ xa theo 1 hướng (Δx > mov_x cho swap phải). mov = 0.6 × kích thước tile. Nghĩa là phải kéo hơn nửa tile mới trigger.
   - **Tolerance zone (tol):** Chiều vuông góc phải giữ nhỏ (|Δy| < tol cho swap ngang). tol = 0.3 × kích thước tile. Nghĩa là không được lệch quá 30% tile theo chiều ngang khi đang vuốt dọc.
4. **Origin reset:** Sau mỗi lần swap thành công, điểm bắt đầu được cập nhật thành vị trí hiện tại → cho phép vuốt liên tiếp nhiều tile mà không cần thả rồi nhặt lại.

**So sánh với Huda et al.:**
- Huda dùng tolerance zone cho steering xe lăn (lệch ngang khi muốn đi thẳng = nguy hiểm).
- Chúng ta dùng tolerance zone cho tile swap (lệch dọc khi muốn vuốt ngang = swap sai).
- Cùng nguyên lý, khác ứng dụng.

**Thêm các thành phần temporal:**
- **3-frame debounce:** Chỉ chấp nhận gesture khi 3 frame liên tiếp đồng ý. VD: pinch-pinch-open-pinch → frame "open" bị bỏ qua vì chỉ 1 frame, không đủ 3. Ngăn false triggers từ transient errors.
- **Adaptive cursor:** EMA smoothing với α thay đổi. Tay run nhỏ → smooth mạnh (α=0.4). Tay di chuyển xa → pass-through (α=1.0).

**Nói:**
> "Đây là đóng góp chính thứ hai — mô hình tương tác toán học adapted từ Huda et al. 2025. Thay vì kéo tự do, chúng em dùng swipe có ngưỡng. Để swap tile sang phải, tay phải di chuyển hơn 60% kích thước tile theo trục X, đồng thời giữ lệch theo trục Y dưới 30% — đây là tolerance zone, lọc ra các chuyển động không chủ ý. Sau mỗi swap, origin reset để cho phép vuốt liên tiếp. Kết hợp với debounce 3 frame và adaptive cursor smoothing, mô hình biến phân loại frame-by-frame thành điều khiển game mượt mà."

---

### Slide 11: Deployed System (1.5 phút)

**Nội dung:** Screenshot game, sơ đồ pipeline triển khai.

**Giải thích cho bản thân:**

- **Toàn bộ chạy trên trình duyệt (client-side):**
  - MediaPipe: WebAssembly (WASM) → phát hiện tay không cần server.
  - MLP: ONNX Runtime Web → inference trên trình duyệt.
  - Game: HTML5 Canvas → render puzzle.
- **Không có server nào.** Người dùng mở URL → tải model 227 KB → chơi ngay. Không gửi data đi đâu cả → privacy bảo mật.
- **30+ FPS:** Đủ mượt cho interaction real-time.
- **Gesture mapping:**
  - Pinch → nhặt tile (grab)
  - Open Hand → thả tile (release)
  - Fist giữ 1.5 giây → reset puzzle (dwell activation ngăn reset vô tình)
  - Frame (2 tay) → chụp ảnh từ webcam làm puzzle

**Nói:**
> "Hệ thống deploy hoàn toàn trên trình duyệt. MediaPipe chạy bằng WebAssembly, MLP chạy bằng ONNX Runtime Web. Tổng model size 227 kilobyte, đạt 30+ FPS, không cần server. Người dùng mở trang web, bật camera, và chơi ngay. Đây là game screenshot — có skeleton overlay, timer, và hướng dẫn cử chỉ."

---

### Slide 12: Conclusion (1.5 phút)

**Nội dung:** 3 contributions + 3 limitations + 3 future work.

**Nói:**
> "Tóm lại, ba đóng góp chính. Thứ nhất: multi-modal comparison cho thấy ceiling effect — skeleton alone đủ cho 5 coarse gestures. Thứ hai: mô hình tương tác adapted từ Huda et al. biến classification thành game control đáng tin cậy. Thứ ba: deploy 100% client-side, 227 KB, 30+ FPS.
>
> Về hạn chế: 5 gestures quá đơn giản — ceiling effect dự kiến sẽ biến mất với vocabulary lớn hơn. Chưa có user study chính thức. Và threshold cố định, chưa calibrate per-user.
>
> Hướng phát triển: mở rộng lên full HaGRID 18 class nơi fusion thực sự cần thiết, calibrate threshold theo từng người, và thực hiện đánh giá usability chính thức."

---

### Slide 13: Thank You + Demo (1 phút)

**Nội dung:** QR code (cv.hieudinh.dev), GitHub repo URL, tên nhóm.

**Nói:**
> "Cảm ơn thầy/cô và các bạn đã lắng nghe. QR code trên màn hình dẫn đến demo trực tiếp tại cv.hieudinh.dev. Code và report có trên GitHub. Mời thầy/cô quét QR để thử game, hoặc chúng em xin demo trực tiếp."

→ **Mở game demo.**

---

## Chuẩn bị Q&A — Bách khoa toàn thư

> **Nguyên tắc:** (1) Acknowledge câu hỏi, (2) Trả lời bằng CON SỐ, (3) Giải thích tại sao, (4) Nếu là hạn chế → thừa nhận + nêu cách khắc phục.

---

### 🔬 METHODOLOGY & EVALUATION

**Q: 100% accuracy — data leakage?**
> Không. GroupKFold với subject ID — không ai ở cả train lẫn test. Per-fold CNN: 99.98%, 99.94%, 100%, 99.98%, 99.94% → không phải tất cả perfect → loại trừ bug. t-SNE: clusters tách biệt. Full HaGRID 18 class SOTA chỉ 95.5% → task dễ, không phải model thần kỳ.

**Q: Tại sao accuracy mà không F1?**
> Báo cáo CẢ HAI. F1 per-class cho thấy "None" yếu nhất (RF: 0.87). Accuracy là overview, F1 là chi tiết. Dataset imbalance 3:1 nhưng không quá nghiêm trọng.

**Q: Person-aware CV là gì?**
> GroupKFold theo subject ID. 13,250 người ÷ 5 folds → test trên ~2,650 người MỚI hoàn toàn. Random split có thể cho cùng 1 người vào train+test → model nhớ mặt/da thay vì học cử chỉ → accuracy giả.

**Q: 5 class quá ít. Ý nghĩa?**
> ĐÚNG là đơn giản. Đó chính xác là point: phát hiện ceiling effect → xác định boundary: dưới 5 geometrically distinct gestures, single-modality MLP đủ. Full HaGRID 18 class = nơi multi-modal cần thiết.

**Q: RF 24 features vs MLP 60D — so sánh không công bằng?**
> Chủ đích. Cho thấy bước nhảy feature engineering → representation learning. RF trên 24 hand-crafted = 94%, MLP trên raw 60D = 99.8%. Narrative: "end-to-end learning tốt hơn domain-specific features."

**Q: YOLO dùng mAP, model khác dùng accuracy?**
> Metrics khác vì bài toán khác: YOLO = detection (tìm VỊ TRÍ + class), MLP = classification (biết vị trí, chỉ phân loại). Không claim YOLO kém hơn — claim YOLO thêm overhead localization không cần (MediaPipe đã detect tay).

**Q: Heuristic 1% sao thấp vậy?**
> Rules thiết kế cho 3D lý tưởng, nhưng MediaPipe trả 2D chiếu từ 3D → mất depth info. Ngón nghiêng về camera → tọa độ 2D giống ngón thẳng. 20 if-else không robust. Bằng chứng: rules-based approach không đủ cho vision.

---

### 🏗️ DESIGN DECISIONS

**Q: CNN 100%, MLP 99.8% — sao không deploy CNN?**
> CNN cần crop 224×224 mỗi frame + 0.3 MB model. MLP cần 60 số (MediaPipe cho sẵn) + 227 KB. Nhỏ hơn, nhanh hơn. 0.2% chênh lệch không có ý nghĩa thống kê (5 folds). Browser 30 FPS → MLP là pragmatic.

**Q: Frame bypass MLP — deployed system là 4-class?**
> Deployed: đúng, MLP 4-class + two-hand rule cho Frame. Benchmarking: MLP eval 5-class. Bypass = practical optimization: "Frame" CẦN 2 tay → detect 2 tay là thuộc tính BẢN CHẤT, đáng tin hơn classify từ crop 1 tay.

**Q: Architecture MLP 60→128→64→5 — tại sao?**
> Thử 1/2/3 layers. 3 layers = sweet spot: đủ capacity cho 5 classes, 14.7K params với dropout 0.3. Deeper → không cải thiện (ceiling effect ngay từ 2 layers). ReLU + BatchNorm, converge ~50 epochs.

**Q: "Vision-First Multi-Modal" nhưng deployed là unimodal?**
> Title = PHƯƠNG PHÁP NGHIÊN CỨU. Vision-first = thử tín hiệu rẻ nhất (skeleton) trước, test thêm modality (appearance) → kết quả: không cần. Multi-modal investigation IS contribution. Deployed unimodal = KẾT LUẬN của multi-modal study.

**Q: Fusion quá đơn giản (weighted average)?**
> Mục tiêu: prove/disprove 2 modalities complement nhau. Late fusion (α=0.7 skeleton) = cách đơn giản nhất. 100% = bằng components → KHÔNG complementary info. Complex fusion (attention) sẽ không cải thiện vì inputs đã saturated. Simple trước, phức tạp chỉ khi cần.

---

### 📐 MATHEMATICAL MODEL

**Q: Chỉ là dead-zone filter — đóng góp gì?**
> Toán đơn giản, đóng góp ở ADAPTATION: Huda dùng tolerance cho continuous steering xe lăn. Chúng em adapt cho discrete grid swap + origin-reset + debounce. Kết hợp biến 99.8% → usable control.

**Q: mov=0.6, tol=0.3 — chọn sao?**
> Empirical: thử 0.4-0.7. 0.6 × tile_size = đủ xa lọc jitter, đủ gần dễ vuốt. tol=0.3: nhỏ quá → khó vuốt thẳng, lớn quá → cho phép vuốt chéo. Hạn chế: fixed, chưa per-user. Future: adaptive theo hand size.

**Q: Debounce 3 frame — tại sao 3?**
> 30 FPS → 3 frame = 100ms. Đủ lọc transient (1-2 frame sai), đủ ngắn user không thấy delay. 5 frame = 167ms → noticeable. 10 = 333ms → sluggish.

**Q: Adaptive cursor — α thay đổi theo gì?**
> α = min(1.0, Δd / threshold). Δd = khoảng cách tay hiện tại vs trước. Tay đứng yên → α nhỏ → smooth mạnh. Tay di xa → α→1.0 → pass-through. Threshold = 30px (empirical).

**Q: FSM có bao nhiêu states?**
> 5: IDLE → HOVERING (tay detected) → GRABBING (pinch) → DRAGGING (di tile) → RELEASED (open hand). Thêm FIST_HOLD (đếm 1.5s dwell → reset). Transitions = gesture output + debounce.

---

### 🌐 DEPLOYMENT

**Q: 30+ FPS trên phần cứng nào?**
> MacBook Pro M1 (Chrome), PC i5 + RTX 3060. Bottleneck = MediaPipe (~20ms), MLP = 0.5ms. Chưa test low-end. MediaPipe WASM có optimizations đa platform.

**Q: "No server" — ai host trang web?**
> Static hosting (GitHub Pages, Netlify). Không server-side processing. Toàn bộ compute trên browser. Server chỉ serve HTML/JS/ONNX. Không gửi camera data → privacy hoàn toàn.

**Q: Model 227 KB — nén thêm được?**
> Quantize FP32→INT8 → ~57 KB. Pruning giảm nữa. Nhưng 227 KB < 1 ảnh JPEG → không cần optimize thêm.

---

### 📊 DATASET

**Q: HaGRID là gì? License?**
> SberDevices (Sber, Nga). CC-BY-SA 4.0. WACV 2024. 554K ảnh, 37,583 subjects, 18 classes. Subset 5 class.

**Q: Normalize landmarks thế nào?**
> Bỏ wrist (gốc tọa độ). 20 landmarks × 3D = 60D. Translate: trừ wrist → hand-centered. Scale: chia max distance → size-invariant. Không rotate — rotation = feature hữu ích.

**Q: Data augmentation?**
> Skeleton: random rotation ±15°, scale ±10%, Gaussian noise σ=0.01. CNN: flip, rotation, color jitter, crop. MLP với augmentation converge nhanh hơn.

---

### ⚔️ CÂU HỎI KHÓ

**Q: Đây là bài tập, "contribution" nghe quá?**
> Dùng framing academic để rèn kỹ năng. "Contribution" = những gì TỰ LÀM (benchmark 6 methods, adapt math model, deploy game) thay vì re-implement tutorial. Không claim novelty.

**Q: MediaPipe là black box — nếu sai?**
> Failure modes: (1) không detect → game đứng yên (fail-safe), (2) detect sai → landmarks noise → MLP có thể sai → debounce giảm thiểu. Acknowledged dependency.

**Q: So sánh Huda — họ giải bài khó hơn (xe lăn)?**
> Đúng, Huda critical hơn. Chúng em borrow PHƯƠNG PHÁP, không claim cùng impact. Wheelchair = continuous control, puzzle = discrete actions. Cả hai xác nhận: "accuracy đơn lẻ ≠ reliable control."

**Q: Không user study — sao claim model works?**
> LIMITATION lớn nhất. Evidence qualitative: development free-drag gây jitter, threshold loại bỏ. Live demo. Formal cần: N users, task completion rate, SUS questionnaire. Future work #1.

**Q: Nếu 5 gestures dễ — tại sao không nhiều hơn?**
> Vocabulary cho GAME (grab, release, reset, capture, idle). Thêm gestures không có action → vô nghĩa. RQ = "cần bao nhiêu complexity cho ứng dụng thực tế?" → 227 KB MLP.

---

## Thuật ngữ nhanh

| Thuật ngữ | 1 câu giải thích |
|-----------|------------------|
| **Multi-modal** | 2+ loại input: skeleton (60D) + appearance (224×224 RGB) |
| **Late fusion** | Weighted average softmax outputs: α=0.7 skel + 0.3 app |
| **Ceiling effect** | Cả 2 modalities ~100%, fusion = 0 improvement |
| **Person-aware CV** | GroupKFold(5, groups=subject_id), 13,250 người |
| **Tolerance zone** | |Δy| < 0.3 × tile khi vuốt ngang → lọc vuốt chéo |
| **Debounce** | 3 frame agree mới chấp nhận → lọc transient errors |
| **EMA cursor** | x = α×x_new + (1-α)×x_old, α adaptive theo speed |
| **FSM** | IDLE→HOVERING→GRABBING→DRAGGING→RELEASED |
| **ONNX** | PyTorch → ONNX → ONNX Runtime Web (browser inference) |
| **WASM** | WebAssembly — MediaPipe native performance trên browser |
| **mAP@50** | % objects detected đúng với IoU > 50% |
| **Dwell activation** | Fist hold 1.5s → reset (tránh reset vô tình) |
| **HaGRID** | HAnd Gesture Recognition IDentification, WACV 2024 |

---

## Phân chia nói

| Phần | Slides | Người | Thời gian |
|------|--------|-------|-----------|
| Mở đầu + Dataset | 1-3 | Người 1 | ~3 phút |
| Methods + Ceiling | 4-8 | Người 2 | ~5 phút |
| Interaction + Deploy | 9-12 | Người 3 | ~5.5 phút |
| Demo + Q&A | 13 | Cả nhóm | ~1.5 phút |

---

## Checklist trước khi nói

- [ ] Test game (bật webcam, chơi thử, verify gesture recognition)
- [ ] PDF presentation full screen
- [ ] Nhớ: 99.8%, 227 KB, 14.7K params, 30+ FPS, 13,250 subjects
- [ ] Ceiling effect 1 câu: "5 cử chỉ quá khác nhau → model nào cũng đúng → fusion vô nghĩa"
- [ ] Math model 1 câu: "vuốt đủ xa + giữ thẳng + 3 frame đồng ý → mới swap"
- [ ] Biết tên paper: Huda et al. 2025, HaGRID v2 (WACV 2024)



