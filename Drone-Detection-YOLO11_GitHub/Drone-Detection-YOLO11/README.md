# Drone Detection using YOLO11

## Projekt

Dieses Projekt wurde im Modul **Maschinelles Sehen** erstellt. Ziel ist die Erkennung von Drohnen mit YOLO11. Zusätzlich zur Standardauswertung mit Precision, Recall und mAP wird untersucht, wie häufig Vögel, Hubschrauber oder Flugzeuge fälschlicherweise als Drohne erkannt werden.

Das Modell ist ein Ein-Klassen-Detektor:

```text
0 = drone
```

## Projektstruktur

```text
Drone-Detection-YOLO11/
├── README.md
├── requirements.txt
├── .gitignore
├── ABGABE_CHECKLISTE.md
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── create_combined_dataset.py
│   └── evaluate_model.py
├── datasets/README.md
├── models/
├── evaluation_results/
└── poster/
```

Die großen Datensätze und automatisch erzeugten Trainingsordner werden nicht im Repository gespeichert. Hinweise stehen in [`datasets/README.md`](datasets/README.md).

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Ablauf

### 1. Datensatz vorbereiten

Die heruntergeladenen Ausgangsdaten werden in `Database1/` abgelegt. Danach:

```bash
python scripts/prepare_dataset.py
```

Aufteilung:

```text
80 % Training
10 % Validation
10 % Test
```

### 2. Modell trainieren

```bash
python scripts/train.py
```

Standardkonfiguration:

```text
Modell: yolo11n.pt
Epochen: 50
Bildgröße: 640 × 640
Batchgröße: 16
```

### 3. Kombinierten Testdatensatz erzeugen

```bash
python scripts/create_combined_dataset.py
```

Der kombinierte Datensatz enthält Testbilder aus beiden Datensätzen, aber ausschließlich Drohnen-Labels der Klasse `0`. Bilder ohne Drohne erhalten leere Labeldateien. Er wird für die offizielle YOLO-Auswertung verwendet.

### 4. Modell auswerten

Vor dem Start muss der Pfad `MODEL_PATH` in `scripts/evaluate_model.py` geprüft werden.

```bash
python scripts/evaluate_model.py
```

## Evaluierung

### Standard-YOLO-Auswertung

Ultralytics berechnet unter anderem:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95
- Precision-Recall-Kurve
- F1-Kurve
- Confusion Matrix

Diese Auswertung beantwortet: **Wie gut erkennt das Modell Drohnen?**

### Eigene Fehlklassifikationsanalyse

Das eigene Skript ermittelt pro tatsächlicher Klasse:

- Anzahl der Testbilder
- Anzahl der als Drohne erkannten Bilder
- Erkennungsrate in Prozent
- durchschnittliche Confidence
- gespeicherte Beispielbilder mit Vorhersage

Diese Auswertung beantwortet: **Welche Objekte verwechselt das Modell mit einer Drohne und wie häufig?**

## Ergebnisse

```text
evaluation_results/
├── image_results.csv
├── class_statistics.csv
├── yolo_statistics.csv
├── plots/
├── detected_as_drone/
└── yolo_evaluation/
```

## Grenzen

- Das Modell kennt nur die Klasse `drone`.
- Vögel, Hubschrauber und Flugzeuge werden nicht als eigene Modellklassen vorhergesagt.
- Die Zusatzanalyse bewertet Fehlalarme auf Bildebene.
- Ergebnisse hängen von Datenqualität, Confidence-Schwelle und Trainingsparametern ab.

## Mögliche Erweiterungen

- mehr schwierige Negativbeispiele
- Vergleich verschiedener YOLO11-Modellgrößen
- Hyperparameteroptimierung
- Mehrklassenmodell
- Video-, Echtzeit- und Tracking-Anwendung

## Referenzen

[1] J. Redmon, S. Divvala, R. Girshick und A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 779–788. doi: 10.1109/CVPR.2016.91.

[2] Ultralytics, “YOLO11,” *Ultralytics Documentation*. Verfügbar: https://docs.ultralytics.com/models/yolo11/ (Zugriff: 30. Juli 2026).

[3] Ultralytics, “Ultralytics YOLO,” *GitHub Repository*. Verfügbar: https://github.com/ultralytics/ultralytics (Zugriff: 30. Juli 2026).

[4] SSHIKAMARU, “Drone Object Detection,” *Kaggle Dataset*. Verfügbar: https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection (Zugriff: 30. Juli 2026).

[5] Roboflow, “Roboflow Universe,” *Dataset Platform*. Verfügbar: https://universe.roboflow.com/ (Zugriff: 30. Juli 2026).

[6] J. D. Hunter, “Matplotlib: A 2D Graphics Environment,” *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007. doi: 10.1109/MCSE.2007.55.

[7] W. McKinney, “Data Structures for Statistical Computing in Python,” in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56–61. doi: 10.25080/Majora-92bf1922-00a.

## Hinweis

Die verwendeten Datensätze unterliegen den jeweiligen Nutzungsbedingungen und werden deshalb nicht direkt mitgeliefert.
