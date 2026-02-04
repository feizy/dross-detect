import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO, RTDETR

def cv_imread(file_path):
    """Read image with Unicode path support."""
    stream = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

def apply_clahe(img):
    """Apply CLAHE to enhance contrast in grayscale/low-contrast images."""
    # Convert to LAB to apply CLAHE on L channel if color, or just apply if grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

def predict_large_image(model, img_path, output_path, stride=512, slice_size=640, use_clahe=True):
    img = cv_imread(str(img_path))
    if img is None:
        return
    
    if use_clahe:
        img_for_pred = apply_clahe(img)
    else:
        img_for_pred = img

    H, W = img.shape[:2]
    all_detections = []
    
    def _safe_start(pos, max_len, win):
        if max_len <= win: return 0
        return min(pos, max_len - win)

    x_steps = int(np.ceil((W - slice_size) / stride)) + 1 if W > slice_size else 1
    y_steps = int(np.ceil((H - slice_size) / stride)) + 1 if H > slice_size else 1
    
    for i in range(y_steps):
        for j in range(x_steps):
            x1 = _safe_start(j * stride, W, slice_size)
            y1 = _safe_start(i * stride, H, slice_size)
            x2, y2 = x1 + slice_size, y1 + slice_size
            
            crop = img_for_pred[y1:y2, x1:x2]
            if crop.shape[0] < slice_size or crop.shape[1] < slice_size:
                crop = cv2.copyMakeBorder(crop, 0, max(0, slice_size - crop.shape[0]), 
                                         0, max(0, slice_size - crop.shape[1]),
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # Predict
            results = model(crop, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    all_detections.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, conf, cls])
    
    if not all_detections:
        cv2.imwrite(str(output_path), img)
        return

    # NMS
    all_detections = np.array(all_detections)
    boxes_xywh = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in all_detections]
    scores = [float(d[4]) for d in all_detections]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.25, nms_threshold=0.45)
    
    # Draw (always on original image source)
    for idx in indices:
        i = int(idx[0]) if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
        x, y, w, h = boxes_xywh[i]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 3)
        cv2.putText(img, f"Dross {scores[i]:.2f}", (int(x), int(y)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        im_buf_arr.tofile(str(output_path))
    print(f"Saved result to {output_path}")

def run_inference(model_type='rtdetr', model_weights=None):
    project_root = Path(__file__).resolve().parents[1]
    
    if model_weights is None:
        # Auto-find latest or use default
        model_weights = 'rtdetr-l.pt' if model_type == 'rtdetr' else 'yolov8s.pt'
    
    if model_type == 'rtdetr':
        model = RTDETR(model_weights)
    else:
        model = YOLO(model_weights)
        
    print(f"Using {model_type} model with weights: {model_weights}")
    
    test_dir = project_root / "有焊渣图片" / "frame" 
    out_dir = project_root / f"inference_{model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    for img_path in images[:10]:
        predict_large_image(model, img_path, out_dir / img_path.name, use_clahe=True)

if __name__ == "__main__":
    # Example: run_inference(model_type='rtdetr', model_weights='path/to/best.pt')
    run_inference(model_type='rtdetr')
