from datetime import datetime
from pathlib import Path
from ultralytics import RTDETR

def train():
    # Load a pretrained RT-DETR model
    # rtdetr-l is a good balance between speed and accuracy
    # You can also try rtdetr-x for higher accuracy
    model = RTDETR('rtdetr-l.pt') 

    # Project root directory
    project_root = Path(__file__).resolve().parents[1]
    
    # Train the model
    # Using the same dross.yaml as YOLOv8
    # imgsz=640 matches the sliced tile size
    results = model.train(
        data=str(project_root / 'dross.yaml'),
        epochs=100,
        imgsz=640,
        batch=8, # RT-DETR requires more GPU memory than YOLOv8s, adjust batch size if needed
        project=str(project_root / 'runs' / 'detect'),
        name='dross_rtdetr_l_'+datetime.now().strftime('%Y%m%d_%H%M'),
        # RT-DETR specifically benefits from these
        mosaic=1.0,
        mixup=0.2, # Often helpful for transformer-based models
        copy_paste=0.1,
        degrees=10.0,
        fliplr=0.5,
        save=True,
    )
    
    print("RT-DETR Training Completed.")

if __name__ == "__main__":
    train()
