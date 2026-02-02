
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Sliding window inference
def cv_imread(file_path):
    stream = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

def predict_large_image(model, img_path, output_path, stride=512, slice_size=640):
    img = cv_imread(str(img_path))
    if img is None:
        return
    
    H, W = img.shape[:2]
    
    # List to store all detections [x1, y1, x2, y2, conf, cls] relative to original image
    all_detections = []
    
    x_steps = int(np.ceil((W - slice_size) / stride)) + 1
    y_steps = int(np.ceil((H - slice_size) / stride)) + 1
    
    for i in range(y_steps):
        for j in range(x_steps):
            x1 = j * stride
            y1 = i * stride
            
            if x1 + slice_size > W: x1 = W - slice_size
            if y1 + slice_size > H: y1 = H - slice_size
            
            x2 = x1 + slice_size
            y2 = y1 + slice_size
            
            crop = img[y1:y2, x1:x2]
            
            # Predict
            results = model(crop, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # coords in crop
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # map back to original
                    ox1 = bx1 + x1
                    oy1 = by1 + y1
                    ox2 = bx2 + x1
                    oy2 = by2 + y1
                    
                    all_detections.append([ox1, oy1, ox2, oy2, conf, cls])
    
    # NMS (Non-Maximum Suppression) on the combined detections
    if not all_detections:
        print(f"No dross detected in {img_path.name}")
        # Save original if no detection or just skip? Let's save original
        cv2.imwrite(str(output_path), img)
        return

    all_detections = np.array(all_detections)
    
    # Apply NMS
    # We use cv2.dnn.NMSBoxes or torch torchvision.ops.nms
    # Here we can use a simple custom one or converting to boxes for opencv
    boxes_xywh = []
    scores = []
    
    for d in all_detections:
        x1, y1, x2, y2, conf, c = d
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([x1, y1, w, h])
        scores.append(float(conf))
        
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.25, nms_threshold=0.45)
    
    # Draw results
    for idx in indices:
        i = idx # opencv return format might vary, sometimes it's [i]
        box = boxes_xywh[i]
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        
        # Draw red rect
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(img, f"Dross {scores[i]:.2f}", (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # cv2.imwrite doesn't support unicode paths on Windows either
    # encoding to stream then to file
    is_success, im_buf_arr = cv2.imencode(".jpg", img)
    if is_success:
        im_buf_arr.tofile(str(output_path))
    print(f"Saved result to {output_path}")

def run_inference():
    model_path = r'e:/code/dross-detect/runs/detect/dross_v8n/weights/best.pt'
    # Fallback to yolov8n.pt if not trained yet, just for testing the script
    if not os.path.exists(model_path):
        print("Model weights not found, using default yolov8n.pt for demo")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)
    
    test_dir = Path(r"e:/code/dross-detect/有焊渣图片/cover") 
    out_dir = Path(r"e:/code/dross-detect/inference_results")
    out_dir.mkdir(exist_ok=True)
    
    # Process some images from cover
    images = [test_dir / "cover10.png", test_dir / "cover104.png", test_dir / "cover105.png"]
    for img_p in images:
        predict_large_image(model, img_p, out_dir / img_p.name)

if __name__ == "__main__":
    run_inference()
