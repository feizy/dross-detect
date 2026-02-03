
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model (small version)

    # Train the model
    # We use the generated dross.yaml
    # img=640 because we sliced them to 640
    project_root = Path(__file__).resolve().parents[1]
    results = model.train(
        data=str(project_root / 'dross.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,
        project=str(project_root / 'runs' / 'detect'),
        name='dross_v8s_smallobj_'+datetime.now().strftime('%Y%m%d_%H%M'),#name最后加上时间
        # small-object friendly augments
        mosaic=1.0,
        copy_paste=0.1,
        close_mosaic=10,
        scale=0.3,
        fliplr=0.5,
        degrees=10.0,
    )
    
    print("Training Completed.")

if __name__ == "__main__":
    train()
