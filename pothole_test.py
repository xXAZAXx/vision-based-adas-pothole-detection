#Testing trained pothole model
from ultralytics import YOLO
import cv2

def main():
    model = YOLO("runs/detect/train9/weights/best.pt")

    # Start webcam inference
    results = model.predict(
        source=0,        # webcam
        conf=0.3,
        device=0,        # GPU
        stream=True
    )

    # IMPORTANT: iterate over results to keep window alive
    for r in results:
        frame = r.plot()   # draw boxes
        cv2.imshow("Pothole Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
