
import os
import cv2
import shutil
import yaml
import numpy as np
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
import random

def cv_imread(file_path):
    """Read image with unicode path support."""
    stream = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

def convert_box_to_shapely(yolo_box, img_w, img_h):
    """Convert YOLO format (cls, x_c, y_c, w, h) to Shapely box."""
    # yolo_box: [class_id, x_c, y_c, w, h] (normalized)
    class_id, x_c, y_c, w, h = yolo_box
    
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return box(x1, y1, x2, y2), int(class_id)

def slice_image_and_labels(
    img_path, 
    label_path, 
    out_img_dir, 
    out_label_dir, 
    slice_size=640, 
    overlap_ratio=0.2
):
    """
    Slice a large image and its corresponding YOLO labels.
    """
    img = cv_imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return

    H, W = img.shape[:2]
    
    # Read labels
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    boxes.append(parts)

    stride = int(slice_size * (1 - overlap_ratio))
    
    tile_count = 0
    
    # Calculate grid
    x_steps = int(np.ceil((W - slice_size) / stride)) + 1
    y_steps = int(np.ceil((H - slice_size) / stride)) + 1

    for i in range(y_steps):
        for j in range(x_steps):
            x1 = j * stride
            y1 = i * stride
            
            # Adjust last tile to fit processed image completely if needed
            # (Or just let it overlap more)
            if x1 + slice_size > W:
                x1 = W - slice_size
            if y1 + slice_size > H:
                y1 = H - slice_size
            
            x2 = x1 + slice_size
            y2 = y1 + slice_size
            
            tile_rect = box(x1, y1, x2, y2)
            
            # Find boxes in this tile
            tile_boxes = []
            for b in boxes:
                b_poly, cls_id = convert_box_to_shapely(b, W, H)
                
                if tile_rect.intersects(b_poly):
                    intersection = tile_rect.intersection(b_poly)
                    # Use Intersection over Box area to decide if we keep it
                    # Smaller fragments might be ignored or kept. 
                    # For small dross, we want to keep it even if partially cut.
                    # But usually we want the box to be mostly inside.
                    
                    # Convert intersection back to YOLO format relative to tile
                    minx, miny, maxx, maxy = intersection.bounds
                    
                    # Clip to tile bounds (already done by intersection, but strictly rel to tile)
                    new_x1 = max(0, minx - x1)
                    new_y1 = max(0, miny - y1)
                    new_x2 = min(slice_size, maxx - x1)
                    new_y2 = min(slice_size, maxy - y1)
                    
                    # Check if box is valid
                    if (new_x2 - new_x1) > 2 and (new_y2 - new_y1) > 2:
                        # Convert to normalized YOLO
                        n_w = (new_x2 - new_x1) / slice_size
                        n_h = (new_y2 - new_y1) / slice_size
                        n_xc = (new_x1 + new_x2) / 2 / slice_size
                        n_yc = (new_y1 + new_y2) / 2 / slice_size
                        
                        tile_boxes.append(f"{cls_id} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")

            # Decide whether to save this tile
            # Strategy: 
            # 1. Save if it contains labels (Positive sample)
            # 2. Save a percentage of empty tiles (Negative sample) for background balance
            
            has_labels = len(tile_boxes) > 0
            should_save = has_labels or (random.random() < 0.1) # Keep 10% of background tiles
            
            if should_save:
                tile_name = f"{img_path.stem}_{i}_{j}.jpg"
                
                # Save Label
                if has_labels: # Only create label file if there are labels? YOLOv8 handles empty files as background images.
                    with open(out_label_dir / f"{tile_name.replace('.jpg', '.txt')}", 'w') as f:
                        f.write("\n".join(tile_boxes))
                
                # Save Image
                subset = img[y1:y2, x1:x2]
                cv2.imwrite(str(out_img_dir / tile_name), subset)
                
                tile_count += 1
                
    return tile_count

def process_dataset(source_dir, output_dir, train_ratio=0.8):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Setup directories
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            (output_dir / "datasets" / kind / split).mkdir(parents=True, exist_ok=True)
            
    images = list(source_dir.rglob("*.png")) + list(source_dir.rglob("*.jpg"))
    labeled_images = [img for img in images if img.with_suffix('.txt').exists()]
    
    print(f"Found {len(images)} total images, {len(labeled_images)} labeled images.")
    
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * train_ratio)
    train_imgs = labeled_images[:split_idx]
    val_imgs = labeled_images[split_idx:]
    
    print(f"Processing... Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    for split, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
        img_out = output_dir / "datasets" / "images" / split
        lbl_out = output_dir / "datasets" / "labels" / split
        
        for img_path in tqdm(img_list, desc=f"Generating {split} set"):
            # Check if corresponding label file exists
            # We assume label file is same name but .txt, in same folder
            lbl_path = img_path.with_suffix('.txt')
            
            # If label file doesn't exist, we skip slicing if we ONLY want labeled data,
            # BUT for training detection, we also want background images.
            # However, if we don't have ANY labels yet, this script won't produce any positive samples.
            
            slice_image_and_labels(img_path, lbl_path, img_out, lbl_out)

    # Generate data.yaml
    yaml_content = {
        'path': str(output_dir.absolute() / 'datasets'),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'dross'}
    }
    
    with open(output_dir / 'dross.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Done. Dataset prepared at {output_dir / 'datasets'}")
    print(f"Config saved to {output_dir / 'dross.yaml'}")

if __name__ == "__main__":
    # Source directory containing '有焊渣图片' (assumed to contain images and eventually .txt labels)
    # The user said they will label '有焊渣图片'.
    # Note: The subfolders are 'cover' and 'frame'. We should handle recursion.
    SOURCE_DIR = r"e:/code/dross-detect/有焊渣图片" 
    OUTPUT_DIR = r"e:/code/dross-detect"
    
    process_dataset(SOURCE_DIR, OUTPUT_DIR)
