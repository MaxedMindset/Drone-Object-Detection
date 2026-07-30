from pathlib import Path
from ultralytics import YOLO

DATASET_CONFIG = Path("drone_dataset/data.yaml")
BASE_MODEL = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16


def main():
    if not DATASET_CONFIG.exists():
        raise FileNotFoundError("drone_dataset/data.yaml wurde nicht gefunden.")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATASET_CONFIG),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="runs/detect",
        name="drone_test_50"
    )


if __name__ == "__main__":
    main()
