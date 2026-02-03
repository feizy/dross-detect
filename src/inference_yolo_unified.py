from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO


def cv_imread(file_path):
    stream = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


def predict_large_image(model, img_path, output_path, stride=512, slice_size=640):
    img = cv_imread(str(img_path))
    if img is None:
        return

    H, W = img.shape[:2]
    all_detections = []

    def _safe_start(pos, max_len, win):
        if max_len <= win:
            return 0
        return min(pos, max_len - win)

    x_steps = int(np.ceil((W - slice_size) / stride)) + 1
    y_steps = int(np.ceil((H - slice_size) / stride)) + 1
    if W <= slice_size:
        x_steps = 1

    for i in range(y_steps):
        for j in range(x_steps):
            x1 = _safe_start(j * stride, W, slice_size)
            y1 = _safe_start(i * stride, H, slice_size)
            x2 = x1 + slice_size
            y2 = y1 + slice_size

            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < slice_size or crop.shape[1] < slice_size:
                pad_right = max(0, slice_size - crop.shape[1])
                pad_bottom = max(0, slice_size - crop.shape[0])
                crop = cv2.copyMakeBorder(
                    crop, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            results = model(crop, verbose=False)
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    ox1 = bx1 + x1
                    oy1 = by1 + y1
                    ox2 = bx2 + x1
                    oy2 = by2 + y1
                    all_detections.append([ox1, oy1, ox2, oy2, conf, cls])

    if not all_detections:
        cv2.imwrite(str(output_path), img)
        return

    all_detections = np.array(all_detections)
    boxes_xywh = []
    scores = []
    for d in all_detections:
        x1, y1, x2, y2, conf, _ = d
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([x1, y1, w, h])
        scores.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.25, nms_threshold=0.45)

    for idx in indices:
        i = int(idx[0]) if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
        x1, y1, w, h = boxes_xywh[i]
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(
            img, f"Dross {scores[i]:.2f}", (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        im_buf_arr.tofile(str(output_path))


def run_inference():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "runs" / "detect" / "dross_v8s_smallobj" / "weights" / "best.pt"
    if not model_path.exists():
        print("Model weights not found, using default yolov8s.pt for demo")
        model = YOLO("yolov8s.pt")
    else:
        model = YOLO(str(model_path))
    print(f"Model loaded from {model_path}")

    data_root = project_root / "有焊渣图片"
    out_root = project_root / "inference_results" / "unified"
    out_root.mkdir(parents=True, exist_ok=True)

    for subdir in ["cover", "frame"]:
        test_dir = data_root / subdir
        if not test_dir.exists():
            continue
        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
        for img_path in images[:20]:
            predict_large_image(model, img_path, out_dir / img_path.name)


if __name__ == "__main__":
    run_inference()
