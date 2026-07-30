from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/drone_test_50/weights/best.pt")
IMAGE_FOLDER = Path("birds_helicopters.v4i.yolov11/test/images")
LABEL_FOLDER = Path("birds_helicopters.v4i.yolov11/test/labels")
COMBINED_DATASET = Path("combined_test_yolo/data.yaml")
OUTPUT_FOLDER = Path("evaluation_results")
CONFIDENCE_THRESHOLD = 0.25
CLASS_NAMES = {0: "drone", 1: "bird", 2: "helicopter", 3: "plane"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    for path in [MODEL_PATH, IMAGE_FOLDER, LABEL_FOLDER, COMBINED_DATASET]:
        if not path.exists(): raise FileNotFoundError(f"Pfad nicht gefunden: {path}")

    plots = OUTPUT_FOLDER / "plots"
    detected = OUTPUT_FOLDER / "detected_as_drone"
    plots.mkdir(parents=True, exist_ok=True)
    for name in CLASS_NAMES.values(): (detected / name).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    rows = []
    images = sorted(p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    for image in images:
        label = LABEL_FOLDER / f"{image.stem}.txt"
        ids = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id in CLASS_NAMES and class_id not in ids: ids.append(class_id)
        names = [CLASS_NAMES[i] for i in ids]
        prediction = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        found = len(prediction.boxes) > 0
        confidence = max(prediction.boxes.conf.cpu().tolist()) if found else 0.0
        if found:
            for name in names: prediction.save(filename=str(detected / name / image.name))
        rows.append({"image": image.name, "ground_truth": ", ".join(names) or "unlabeled", "drone_detected": found, "number_of_detections": len(prediction.boxes), "max_confidence": round(confidence, 4)})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FOLDER / "image_results.csv", index=False)
    stats = []
    for name in CLASS_NAMES.values():
        part = results[results["ground_truth"].str.contains(name, regex=False)]
        hits = part[part["drone_detected"] == True]
        stats.append({"class": name, "number_of_images": len(part), "detected_as_drone": len(hits), "detection_rate_percent": round(len(hits) / len(part) * 100, 2) if len(part) else 0.0, "average_confidence": round(hits["max_confidence"].mean(), 4) if len(hits) else 0.0})
    statistics = pd.DataFrame(stats)
    statistics.to_csv(OUTPUT_FOLDER / "class_statistics.csv", index=False)

    for column, ylabel, title, filename, limit in [
        ("detection_rate_percent", "Als Drohne erkannt in %", "Erkennungsrate pro Klasse", "detection_rate.png", (0, 100)),
        ("detected_as_drone", "Anzahl der Bilder", "Als Drohne erkannte Bilder", "detected_images.png", None),
        ("average_confidence", "Durchschnittliche Confidence", "Durchschnittliche Confidence pro Klasse", "average_confidence.png", (0, 1))]:
        plt.figure(figsize=(8, 5)); plt.bar(statistics["class"], statistics[column]); plt.xlabel("Tatsächliche Klasse"); plt.ylabel(ylabel); plt.title(title)
        if limit: plt.ylim(*limit)
        plt.tight_layout(); plt.savefig(plots / filename, dpi=300); plt.close()

    metrics = model.val(data=str(COMBINED_DATASET), split="test", conf=CONFIDENCE_THRESHOLD, plots=True, project=str(OUTPUT_FOLDER), name="yolo_evaluation")
    yolo = pd.DataFrame({"metric": ["precision", "recall", "mAP50", "mAP50-95"], "value": [metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map]})
    yolo.to_csv(OUTPUT_FOLDER / "yolo_statistics.csv", index=False)
    plt.figure(figsize=(8, 5)); plt.bar(yolo["metric"], yolo["value"]); plt.ylim(0, 1); plt.title("YOLO-Evaluierung"); plt.tight_layout(); plt.savefig(plots / "yolo_metrics.png", dpi=300); plt.close()
    print("Evaluierung abgeschlossen.")


if __name__ == "__main__":
    main()
