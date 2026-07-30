import random
import shutil
from pathlib import Path

source_dir = Path("Database1")
output_dir = Path("drone_dataset")
random.seed(42)
image_extensions = {".jpg", ".jpeg", ".png"}


def create_directories():
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(image_path, split):
    label_path = image_path.with_suffix(".txt")
    shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
    destination = output_dir / "labels" / split / label_path.name
    if label_path.exists():
        shutil.copy2(label_path, destination)
    else:
        destination.touch()


def main():
    if not source_dir.exists():
        raise FileNotFoundError("Der Ordner Database1 wurde nicht gefunden.")

    create_directories()
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in image_extensions]
    random.shuffle(images)
    train_end = int(len(images) * 0.8)
    val_end = train_end + int(len(images) * 0.1)

    for image in images[:train_end]: copy_pair(image, "train")
    for image in images[train_end:val_end]: copy_pair(image, "val")
    for image in images[val_end:]: copy_pair(image, "test")

    yaml = """path: drone_dataset
train: images/train
val: images/val
test: images/test

names:
  0: drone
"""
    (output_dir / "data.yaml").write_text(yaml, encoding="utf-8")
    print("Datensatz erfolgreich vorbereitet.")


if __name__ == "__main__":
    main()
