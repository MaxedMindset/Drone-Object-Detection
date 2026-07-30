import shutil
from pathlib import Path


# ==========================================================
# Pfade
# ==========================================================

DRONE_IMAGES = Path(r"drone_dataset/images/test")
DRONE_LABELS = Path(r"drone_dataset/labels/test")


EXTERNAL_IMAGES = Path(r"birds_helicopters.v4i.yolov11/test/images")
EXTERNAL_LABELS = Path(r"birds_helicopters.v4i.yolov11/test/labels")

# Ausgabeordner

OUTPUT = Path(r"combined_test_yolo")

IMAGES_OUT = OUTPUT / "images"
LABELS_OUT = OUTPUT / "labels"

# Unterstützte Bildformate

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================================
# Hilfsfunktionen
# ==========================================================

def get_images(folder: Path) -> list[Path]:

    return [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


# Leere YOLO-Labeldatei erstellen

def create_empty_label(destination: Path) -> None:
    destination.write_text("", encoding="utf-8")


# Kopiert ausschließlich Drohnen-Bounding-Boxes (Klasse 0)

def copy_drone_labels(
    source_label: Path,
    destination_label: Path
) -> None:

    if not source_label.exists():
        create_empty_label(destination_label)
        return

    new_lines = []

    with source_label.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 5:
                print(
                    f"Warnung: Ungültige Zeile in "
                    f"{source_label}, Zeile {line_number}"
                )
                continue

            try:
                class_id = int(float(values[0]))
            except ValueError:
                print(
                    f"Warnung: Ungültige Klassen-ID in "
                    f"{source_label}, Zeile {line_number}"
                )
                continue

            # Nur Drohnen übernehmen (Klasse 0)
            if class_id == 0:
                values[0] = "0"
                new_lines.append(" ".join(values))

    destination_label.write_text(
        "\n".join(new_lines),
        encoding="utf-8"
    )


# ==========================================================
# Ausgabeordner erstellen
# ==========================================================

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Bilder laden
# ==========================================================

drone_images = get_images(DRONE_IMAGES)
external_images = get_images(EXTERNAL_IMAGES)

# ==========================================================
# Datensatz 1 kopieren
# ==========================================================

for image_path in drone_images:

    new_stem = f"dataset1_{image_path.stem}"
    new_image_name = new_stem + image_path.suffix.lower()

    destination_image = IMAGES_OUT / new_image_name
    destination_label = LABELS_OUT / f"{new_stem}.txt"

    source_label = DRONE_LABELS / f"{image_path.stem}.txt"

    shutil.copy2(image_path, destination_image)

    copy_drone_labels(
        source_label=source_label,
        destination_label=destination_label
    )


# ==========================================================
# Datensatz 2 kopieren
# ==========================================================

for image_path in external_images:

    new_stem = f"dataset2_{image_path.stem}"
    new_image_name = new_stem + image_path.suffix.lower()

    destination_image = IMAGES_OUT / new_image_name
    destination_label = LABELS_OUT / f"{new_stem}.txt"

    source_label = EXTERNAL_LABELS / f"{image_path.stem}.txt"

    shutil.copy2(image_path, destination_image)

    copy_drone_labels(
        source_label=source_label,
        destination_label=destination_label
    )


# ==========================================================
# data.yaml erstellen
# ==========================================================

yaml_content = f"""path: {OUTPUT.resolve().as_posix()}

train: images
val: images
test: images

names:
  0: drone
"""

yaml_path = OUTPUT / "data.yaml"

yaml_path.write_text(
    yaml_content,
    encoding="utf-8"
)