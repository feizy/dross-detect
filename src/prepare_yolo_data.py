
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

def _choose_slice_params(img_h, img_w, base_slice_size=640):
    """
    Choose slice size and overlap based on image resolution.
    This avoids negative indices for small images and reduces redundant tiles.
    """
    min_side = min(img_h, img_w)
    slice_size = min(base_slice_size, min_side)
    if slice_size <= 480:
        overlap_ratio = 0.10
    elif slice_size <= 640:
        overlap_ratio = 0.20
    else:
        overlap_ratio = 0.20
    return int(slice_size), overlap_ratio

def slice_image_and_labels(
    img_path, 
    label_path, 
    out_img_dir, 
    out_label_dir, 
    base_slice_size=640
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

    tile_count = 0

    def _save_tile(tile_img, tile_boxes, tile_name, target_w, target_h):
        if tile_img.shape[1] < target_w or tile_img.shape[0] < target_h:
            pad_right = max(0, target_w - tile_img.shape[1])
            pad_bottom = max(0, target_h - tile_img.shape[0])
            tile_img = cv2.copyMakeBorder(
                tile_img, 0, pad_bottom, 0, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        has_labels = len(tile_boxes) > 0
        should_save = has_labels or (random.random() < 0.1)
        if not should_save:
            return 0

        if has_labels:
            with open(out_label_dir / f"{tile_name.replace('.jpg', '.txt')}", 'w') as f:
                f.write("\n".join(tile_boxes))

        cv2.imwrite(str(out_img_dir / tile_name), tile_img)
        return 1

    if W < base_slice_size:
        # Small-width images: vertical slicing only, then pad to target size.
        slice_h = min(base_slice_size, H)
        slice_w = W
        overlap_ratio = 0.2
        stride = max(1, int(slice_h * (1 - overlap_ratio)))
        y_steps = int(np.ceil((H - slice_h) / stride)) + 1

        for i in range(y_steps):
            x1 = 0
            y1 = i * stride
            if y1 + slice_h > H:
                y1 = max(0, H - slice_h)

            x2 = x1 + slice_w
            y2 = y1 + slice_h

            tile_rect = box(x1, y1, x2, y2)
            tile_boxes = []
            for b in boxes:
                b_poly, cls_id = convert_box_to_shapely(b, W, H)
                if tile_rect.intersects(b_poly):
                    intersection = tile_rect.intersection(b_poly)
                    minx, miny, maxx, maxy = intersection.bounds
                    new_x1 = max(0, minx - x1)
                    new_y1 = max(0, miny - y1)
                    new_x2 = min(slice_w, maxx - x1)
                    new_y2 = min(slice_h, maxy - y1)

                    if (new_x2 - new_x1) > 2 and (new_y2 - new_y1) > 2:
                        n_w = (new_x2 - new_x1) / base_slice_size
                        n_h = (new_y2 - new_y1) / base_slice_size
                        n_xc = (new_x1 + new_x2) / 2 / base_slice_size
                        n_yc = (new_y1 + new_y2) / 2 / base_slice_size
                        tile_boxes.append(f"{cls_id} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")

            tile_name = f"{img_path.stem}_{i}_0.jpg"
            subset = img[y1:y2, x1:x2]
            tile_count += _save_tile(subset, tile_boxes, tile_name, base_slice_size, base_slice_size)

        return tile_count

    # Normal images: use adaptive slice size and overlap.
    slice_size, overlap_ratio = _choose_slice_params(H, W, base_slice_size=base_slice_size)
    stride = max(1, int(slice_size * (1 - overlap_ratio)))
    
    # Calculate grid
    x_steps = int(np.ceil((W - slice_size) / stride)) + 1
    y_steps = int(np.ceil((H - slice_size) / stride)) + 1

    for i in range(y_steps):
        for j in range(x_steps):
            x1 = j * stride
            y1 = i * stride
            
            # Adjust last tile to fit processed image completely if needed
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
                    minx, miny, maxx, maxy = intersection.bounds
                    
                    new_x1 = max(0, minx - x1)
                    new_y1 = max(0, miny - y1)
                    new_x2 = min(slice_size, maxx - x1)
                    new_y2 = min(slice_size, maxy - y1)
                    
                    if (new_x2 - new_x1) > 2 and (new_y2 - new_y1) > 2:
                        n_w = (new_x2 - new_x1) / slice_size
                        n_h = (new_y2 - new_y1) / slice_size
                        n_xc = (new_x1 + new_x2) / 2 / slice_size
                        n_yc = (new_y1 + new_y2) / 2 / slice_size
                        
                        tile_boxes.append(f"{cls_id} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")

            tile_name = f"{img_path.stem}_{i}_{j}.jpg"
            subset = img[y1:y2, x1:x2]
            tile_count += _save_tile(subset, tile_boxes, tile_name, slice_size, slice_size)

    return tile_count

def _draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_c, y_c, bw, bh = map(float, parts[:5])
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def visualize_slices(dataset_dir, split="train", sample_count=20):
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    out_dir = dataset_dir / "verify" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"No images found for visualization in {img_dir}")
        return

    random.shuffle(images)
    for img_path in images[:sample_count]:
        img = cv_imread(str(img_path))
        if img is None:
            continue
        label_path = lbl_dir / f"{img_path.stem}.txt"
        vis = _draw_yolo_boxes(img, label_path)
        cv2.imwrite(str(out_dir / img_path.name), vis)

def process_dataset(source_dir, output_dir, dataset_name="all", subdirs=None, train_ratio=0.8):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    dataset_dir = output_dir / f"datasets_{dataset_name}"
    
    # Setup directories
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            (dataset_dir / kind / split).mkdir(parents=True, exist_ok=True)
    
    images = []
    if subdirs:
        for subdir in subdirs:
            images += list((source_dir / subdir).rglob("*.png"))
            images += list((source_dir / subdir).rglob("*.jpg"))
    else:
        images = list(source_dir.rglob("*.png")) + list(source_dir.rglob("*.jpg"))
    labeled_images = [img for img in images if img.with_suffix('.txt').exists()]
    
    print(f"Found {len(images)} total images, {len(labeled_images)} labeled images.")
    
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * train_ratio)
    train_imgs = labeled_images[:split_idx]
    val_imgs = labeled_images[split_idx:]
    
    print(f"Processing... Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    for split, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        
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
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'dross'}
    }
    
    yaml_filename = "dross.yaml" if dataset_name == "all" else f"dross_{dataset_name}.yaml"
    yaml_path = output_dir / yaml_filename
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Done. Dataset prepared at {dataset_dir}")
    print(f"Config saved to {yaml_path}")

if __name__ == "__main__":
    # Source directory containing '有焊渣图片' (assumed to contain images and eventually .txt labels)
    # The user said they will label '有焊渣图片'.
    # Note: The subfolders are 'cover' and 'frame'. We should handle recursion.
    project_root = Path(__file__).resolve().parents[1]
    SOURCE_DIR = project_root / "有焊渣图片"
    OUTPUT_DIR = project_root
    
    process_dataset(SOURCE_DIR, OUTPUT_DIR, dataset_name="all", subdirs=None)

    # Optional: visualize a few samples to verify labels after slicing
    VISUALIZE = True
    if VISUALIZE:
        visualize_slices(OUTPUT_DIR / "datasets_all", split="train", sample_count=20)
