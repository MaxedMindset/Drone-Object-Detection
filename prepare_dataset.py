import os
import random
import shutil
from pathlib import Path


# Ordner mit den ursprünglichen Bildern und Labels
source_dir = Path("Database1")

# Neuer YOLO-Datensatz
output_dir = Path("drone_dataset")

# Aufteilung
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Gleiche zufällige Aufteilung bei jedem Start
random.seed(42)


# Erlaubte Bildendungen
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def create_directories():
    """Erstellt die YOLO-Ordner."""

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_images():
    """Sucht alle Bilder im Datensatz."""

    images = []

    for file in source_dir.iterdir():
        if file.suffix in image_extensions:
            images.append(file)

    return images


def copy_pair(image_path, split):
    """Kopiert ein Bild und die zugehörige Labeldatei."""

    label_path = image_path.with_suffix(".txt")

    image_destination = output_dir / "images" / split / image_path.name
    label_destination = output_dir / "labels" / split / label_path.name

    shutil.copy2(image_path, image_destination)

    if label_path.exists():
        shutil.copy2(label_path, label_destination)
    else:
        # Leere Labeldatei für ein negatives Bild erzeugen
        label_destination.touch()
        print("Leeres Label erstellt für:", image_path.name)


def main():
    create_directories()

    images = find_images()
    random.shuffle(images)

    number_of_images = len(images)

    train_end = int(number_of_images * train_ratio)
    val_end = train_end + int(number_of_images * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for image in train_images:
        copy_pair(image, "train")

    for image in val_images:
        copy_pair(image, "val")

    for image in test_images:
        copy_pair(image, "test")

    print()
    print("Datensatz erfolgreich vorbereitet.")
    print("Training:", len(train_images))
    print("Validation:", len(val_images))
    print("Test:", len(test_images))


if __name__ == "__main__":
    main()