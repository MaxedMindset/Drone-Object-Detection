import shutil
from pathlib import Path

DRONE_IMAGES = Path("drone_dataset/images/test")
DRONE_LABELS = Path("drone_dataset/labels/test")
EXTERNAL_IMAGES = Path("birds_helicopters.v4i.yolov11/test/images")
EXTERNAL_LABELS = Path("birds_helicopters.v4i.yolov11/test/labels")
OUTPUT = Path("combined_test_yolo")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_drone_labels(source, destination):
    if not source.exists():
        destination.write_text("", encoding="utf-8")
        return
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 5 and int(float(values[0])) == 0:
            values[0] = "0"
            lines.append(" ".join(values))
    destination.write_text("\n".join(lines), encoding="utf-8")


def copy_dataset(images, labels, prefix):
    for image in images.iterdir():
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            stem = f"{prefix}_{image.stem}"
            shutil.copy2(image, OUTPUT / "images" / f"{stem}{image.suffix.lower()}")
            copy_drone_labels(labels / f"{image.stem}.txt", OUTPUT / "labels" / f"{stem}.txt")


def main():
    for path in [DRONE_IMAGES, DRONE_LABELS, EXTERNAL_IMAGES, EXTERNAL_LABELS]:
        if not path.exists(): raise FileNotFoundError(f"Pfad nicht gefunden: {path}")
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels").mkdir(parents=True, exist_ok=True)
    copy_dataset(DRONE_IMAGES, DRONE_LABELS, "dataset1")
    copy_dataset(EXTERNAL_IMAGES, EXTERNAL_LABELS, "dataset2")
    yaml = f"""path: {OUTPUT.resolve().as_posix()}
train: images
val: images
test: images

names:
  0: drone
"""
    (OUTPUT / "data.yaml").write_text(yaml, encoding="utf-8")
    print("Kombinierter Testdatensatz wurde erstellt.")


if __name__ == "__main__":
    main()
