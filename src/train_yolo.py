
from ultralytics import YOLO
import sys

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (nano version)

    # Train the model
    # We use the generated dross.yaml
    # img=640 because we sliced them to 640
    results = model.train(
        data=r'e:/code/dross-detect/dross.yaml', 
        epochs=10, 
        imgsz=640, 
        batch=16,
        project=r'e:/code/dross-detect/runs/detect',
        name='dross_v8n',
    )
    
    print("Training Completed.")
    print(f"Best model saved at {results.best}")

if __name__ == "__main__":
    train()
