from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")

    model.train(
        data="datasets/Pothole.v1-raw.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        device=1,      # GPU == 1, or CPU == 0 (This is usually the default)
        workers=0      
    )


