from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from ultralytics import YOLO

MODEL_PATH = Path(
    r"runs/detect/runs/drone_test_50/weights/best.pt"
)

IMAGE_FOLDER = Path(
    r"birds_helicopters.v4i.yolov11/test/images"
)

LABEL_FOLDER = Path(
    r"birds_helicopters.v4i.yolov11/test/labels"
)

OUTPUT_FOLDER = Path(
    r"evaluation_results"
)

CONFIDENCE_THRESHOLD = 0.25


# Klassen des externen Testdatensatzes

CLASS_NAMES = {
    0: "drone",
    1: "bird",
    2: "helicopter",
    3: "plane"
}


# Unterstützte Bildformate

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ----------------------------------------------------------
# Ausgabeordner erstellen
# ----------------------------------------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PLOTS_FOLDER = OUTPUT_FOLDER / "plots"
DETECTED_FOLDER = OUTPUT_FOLDER / "detected_as_drone"

PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
DETECTED_FOLDER.mkdir(parents=True, exist_ok=True)


for class_name in CLASS_NAMES.values():

    class_folder = DETECTED_FOLDER / class_name
    class_folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Modell laden
# ----------------------------------------------------------

model = YOLO(MODEL_PATH)


# ----------------------------------------------------------
# Bilder einlesen
# ----------------------------------------------------------

image_paths = []

for path in IMAGE_FOLDER.iterdir():

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths.append(path)


image_paths.sort()


# ----------------------------------------------------------
# Ergebnisse speichern
# ----------------------------------------------------------

results_list = []


# ----------------------------------------------------------
# Bilder auswerten
# ----------------------------------------------------------

for image_path in image_paths:

    label_path = LABEL_FOLDER / f"{image_path.stem}.txt"

    ground_truth_classes = []

    if label_path.exists():

        with label_path.open("r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                class_id = int(values[0])

                if class_id not in ground_truth_classes:
                    ground_truth_classes.append(class_id)


    # Namen der Ground-Truth-Klassen

    ground_truth_names = []

    for class_id in ground_truth_classes:

        class_name = CLASS_NAMES[class_id]
        ground_truth_names.append(class_name)


    # Modellvorhersage

    prediction = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]


    number_of_detections = len(prediction.boxes)

    drone_detected = number_of_detections > 0


    # Höchste Confidence des Bildes bestimmen

    max_confidence = 0.0

    if drone_detected:

        confidences = prediction.boxes.conf.cpu().tolist()
        max_confidence = max(confidences)


    # Annotiertes Bild speichern

    if drone_detected:

        annotated_image = prediction.plot()

        for ground_truth_name in ground_truth_names:

            output_image_path = (
                DETECTED_FOLDER
                / ground_truth_name
                / image_path.name
            )

            prediction.save(
                filename=str(output_image_path)
            )


    # Falls ein Bild keine Labelklasse besitzt

    if not ground_truth_names:
        ground_truth_text = "unlabeled"
    else:
        ground_truth_text = ", ".join(ground_truth_names)


    results_list.append({
        "image": image_path.name,
        "ground_truth": ground_truth_text,
        "drone_detected": drone_detected,
        "number_of_detections": number_of_detections,
        "max_confidence": round(max_confidence, 4)
    })


# ----------------------------------------------------------
# Einzelresultate als CSV speichern
# ----------------------------------------------------------

results_dataframe = pd.DataFrame(results_list)

results_dataframe.to_csv(
    OUTPUT_FOLDER / "image_results.csv",
    index=False
)


# ----------------------------------------------------------
# Statistik pro Klasse berechnen
# ----------------------------------------------------------

statistics_list = []


for class_id, class_name in CLASS_NAMES.items():

    class_rows = results_dataframe[
        results_dataframe["ground_truth"].str.contains(
            class_name,
            regex=False
        )
    ]

    number_of_images = len(class_rows)

    detected_images = class_rows[
        class_rows["drone_detected"] == True
    ]

    number_detected = len(detected_images)


    if number_of_images > 0:

        detection_rate = (
            number_detected
            / number_of_images
            * 100
        )

    else:

        detection_rate = 0.0


    if number_detected > 0:

        average_confidence = (
            detected_images["max_confidence"].mean()
        )

    else:

        average_confidence = 0.0


    statistics_list.append({
        "class": class_name,
        "number_of_images": number_of_images,
        "detected_as_drone": number_detected,
        "detection_rate_percent": round(
            detection_rate,
            2
        ),
        "average_confidence": round(
            average_confidence,
            4
        )
    })


statistics_dataframe = pd.DataFrame(statistics_list)


statistics_dataframe.to_csv(
    OUTPUT_FOLDER / "class_statistics.csv",
    index=False
)


# ----------------------------------------------------------
# Diagramm 1: Erkennungsrate
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    statistics_dataframe["class"],
    statistics_dataframe["detection_rate_percent"]
)

plt.xlabel("Tatsächliche Klasse")
plt.ylabel("Als Drohne erkannt in %")
plt.title("Erkennungsrate pro Klasse")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    PLOTS_FOLDER / "detection_rate.png",
    dpi=300
)

plt.close()


# ----------------------------------------------------------
# Diagramm 2: Erkannte Bilder
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    statistics_dataframe["class"],
    statistics_dataframe["detected_as_drone"]
)

plt.xlabel("Tatsächliche Klasse")
plt.ylabel("Anzahl der Bilder")
plt.title("Als Drohne erkannte Bilder")

plt.tight_layout()

plt.savefig(
    PLOTS_FOLDER / "detected_images.png",
    dpi=300
)

plt.close()


# ----------------------------------------------------------
# Diagramm 3: Durchschnittliche Confidence
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    statistics_dataframe["class"],
    statistics_dataframe["average_confidence"]
)

plt.xlabel("Tatsächliche Klasse")
plt.ylabel("Durchschnittliche Confidence")
plt.title("Durchschnittliche Confidence pro Klasse")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_FOLDER / "average_confidence.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# YOLO-Evaluierung auf dem kombinierten Testdatensatz
# ----------------------------------------------------------

COMBINED_DATASET = Path(
    r"combined_test_yolo/data.yaml"
)


metrics = model.val(
    data=str(COMBINED_DATASET),
    split="test",
    conf=CONFIDENCE_THRESHOLD,
    plots=True,
    project=str(OUTPUT_FOLDER),
    name="yolo_evaluation"
)


yolo_statistics = pd.DataFrame({
    "metric": [
        "precision",
        "recall",
        "mAP50",
        "mAP50-95"
    ],
    "value": [
        metrics.box.mp,
        metrics.box.mr,
        metrics.box.map50,
        metrics.box.map
    ]
})


yolo_statistics.to_csv(
    OUTPUT_FOLDER / "yolo_statistics.csv",
    index=False
)


plt.figure(figsize=(8, 5))

plt.bar(
    yolo_statistics["metric"],
    yolo_statistics["value"]
)

plt.xlabel("Metrik")
plt.ylabel("Wert")
plt.title("YOLO-Evaluierung")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PLOTS_FOLDER / "yolo_metrics.png",
    dpi=300
)

plt.close()