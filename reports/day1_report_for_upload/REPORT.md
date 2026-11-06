# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`="cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** trong ảnh, Vì chỉ có tên lớp, rank và score mà không có bounding box hay mask chỉ vị trí một vật thể. Taxonomy ImageNet-1K cũng là bộ nhãn phân loại cho toàn ảnh, nên rank 1 là lớp mà model cho là phù hợp nhất với cả bức ảnh.
3. Danh sách lớp do con người định nghĩa trước, rồi model được huấn luyện để chọn trong danh sách đó. Lớp cuối của model chỉ xuất ra các con số ứng với từng vị trí class_id; tên lớp có được là nhờ tra class_id trong taxonomy đi kèm checkpoint. Vì vậy model chỉ có thể trả lời bằng các tên có sẵn, không thể đoán ra lớp nằm ngoài danh sách.
4.  `class_id` là số định danh, máy xử lý chính xác, không bị lệch vì cách viết (hoa/thường, số ít/số nhiều, ngôn ngữ).
    `class_name` giúp người đọc hiểu nhanh 468 là gì.
    `taxonomy_name` cho biết class_id và class_name thuộc bộ nhãn nào. Cùng một số id ở taxonomy khác sẽ là lớp khác, và cùng một tên có thể có id khác nhau. Ví dụ 468 chỉ là “cab” trong ImageNet-1K; sang COCO-80 thì số 468 không còn nghĩa đó.

Chỉ ghi tên lớp thì không biết thuộc bộ nhãn nào, khó đối chiếu, dễ nhầm và không tái lập được kết quả.
5. Guideline cần có quy tắc chọn đúng một lớp, ví dụ:

Tiêu chí chọn: chọn vật chính, tức vật lớn nhất, nằm giữa ảnh, hoặc là chủ thể mà ảnh hướng tới.
Thứ tự ưu tiên khi các vật ngang nhau.
Cách xử lý ca mơ hồ: không chắc thì không tự bịa, mà escalate hỏi Lab Coach.
Có ví dụ đúng/sai và quy định chỉ chọn lớp nằm trong taxonomy đã cho.

Nhờ vậy các annotator khác nhau sẽ chọn giống nhau, nhãn nhất quán.
6. Nếu dùng prediction của model làm đáp án rồi đem chấm lại chính model đó thì vòng lặp tự xác nhận: lỗi của model bị coi là đúng, không đo được chất lượng thật.
Score cũng không phải điểm chất lượng nhãn người, hai thứ này đo hai việc khác nhau.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="bowl" ; `score`=0.499948 ; `bbox_xyxy`=[59.29, 288.92, 133.46, 328.98] ; `bbox_width`=74.18 ; `bbox_height`= 40.06
2. Góc trên-trái (59.29, 288.92), góc dưới-phải (133.46, 328.98); rộng khoảng 74.18 px, cao 40.06 px.
3. Ngưỡng 0.20: 17 prediction; ngưỡng 0.35: 11 prediction ;ngưỡng 0.60: 6 prediction.
4. Reviewer phải kiểm tra nhiều hộp hơn, loại hộp sai, xử lý hộp trùng, sửa hộp lệch, escalate ca mơ hồ.
5. Mỗi vật một hộp ôm sát phần nhìn thấy, không chừa thừa, không cắt mất phần nào.
6. person 0.61 bị bàn che và cắt mép trái: viết thành quy tắc guideline vì lặp lại nhiều; chỉ escalate ca guideline chưa nói tới.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. instance_id=kitchen-002 ; class_name=bowl ; score=0.735743 ; polygon_point_count=67 ; polygon_xy đầu = [[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0]]
2. Box là hình chữ nhật có cả nền thừa; polygon 67 điểm đi theo mép thật nên cho biết vùng và dáng chính xác của bowl.
3. instance_id phân biệt từng cá thể trong một ảnh; không phải class ID và không phải tracking ID (theo dõi qua video).
4. Biên polygon đi sát mép phần nhìn thấy của vật, không lấn nền, không cắt vào trong; mỗi vật một polygon.
5. Bowl sát/chồng nhau ở góc trái dưới: dùng guideline (vẽ phần nhìn thấy, tách ranh giới); biên quá mờ hoặc không rõ là vật gì thì escalate Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh (đơn vị nhãn = ảnh), lấy từ taxonomy cho sẵn: `class_id`, `class_name`, `taxonomy_name`. | Record `cab` rank 1 có `score`=0.510915 (~51%), model chưa chắc; record không có vị trí nên không biết "cab" là vật nào. Ảnh nhiều vật thì khó chọn một lớp. | Đọc guideline, chọn một lớp đúng theo tiêu chí (vật chính/chủ thể); ca mơ hồ thì escalate, không tự bịa. | Nhãn có nằm trong taxonomy không, có đúng chủ thể ảnh không, nhiều annotator chọn có giống nhau không. Không lấy `score` làm đáp án. |
| Phát hiện vật thể | Một hộp chữ nhật cho mỗi vật (đơn vị nhãn = hộp): lớp + `bbox_xyxy`. | Ảnh kitchen: ngưỡng 0.20 có 17 hộp, 0.35 có 11 hộp, 0.60 chỉ còn 6 hộp. Các hộp thêm khi hạ ngưỡng (cup, spoon, potted plant, dining table, bottle) score thấp nên dễ sai. `bowl` score 0.499948 nằm sát bowl khác; `person 0.61` bị bàn che và cắt mép trái. | Vẽ một hộp ôm sát phần nhìn thấy của mỗi vật, chọn đúng lớp, không bỏ sót vật nhỏ; vật bị che/cắt mép làm theo guideline. | Hộp có ôm sát không (không thừa, không cắt vào vật), có hộp thừa/trùng/thiếu không, lớp có đúng không (cup hay bowl); duyệt hoặc trả về rework. |
| Instance segmentation | Một polygon/mask cho mỗi cá thể (đơn vị nhãn = từng vật): lớp + `instance_id` + `polygon_xy`. | `kitchen-002` (bowl, score 0.735743) có 67 điểm polygon, bbox [31.44, 343.77, 100.89, 385.55] nằm sát các bowl khác nên ranh giới khó tách; box chứa nền thừa còn polygon đi theo mép cong. | Vẽ biên sát mép phần nhìn thấy, mỗi vật một polygon riêng; hai vật chạm nhau thì vẽ đường ranh giới; biên quá mờ thì escalate. | Biên có lấn ra nền hoặc cắt vào vật không, hai vật dính nhau có bị gộp thành một mask không, `instance_id` có tách đúng từng cá thể không; duyệt hoặc rework. |

## 5. An toàn dữ liệu
1. Chỉ dùng ảnh mẫu do lab cung cấp, không chia sẻ hay đưa ảnh/dữ liệu ra ngoài bài lab, và không để lộ token hay thông tin nhạy cảm (dùng Colab Secrets, không dán vào code).
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình dừng lại và báo cho Lab Coach (escalation), không tự xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
