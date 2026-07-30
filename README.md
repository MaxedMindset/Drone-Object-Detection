Deep Learning basierte Drohnenerkennung in Bildern 

## Projekt

In diesem Projekt wurde ein YOLO11-Modell zur Erkennung von Drohnen trainiert. Ziel war es, Drohnen möglichst zuverlässig zu erkennen und gleichzeitig zu untersuchen, ob ähnliche Objekte wie Vögel, Hubschrauber oder Flugzeuge fälschlicherweise als Drohnen erkannt werden. Für das Training wurde ein Drohnendatensatz verwendet. Anschließend wurde das trainierte Modell mit einem externen Datensatz getestet.

# Projektstruktur

DroneProject
│
├── drone_dataset/
│
├── birds_helicopters.v4i.yolov11/
│
├── combined_test_yolo/
│
├── runs/
│
├── train.py
├── create_combined_dataset.py
├── evaluate_model.py
└── README.md

# Datensätze

## Drohnendatensatz

Der Ordner `drone_dataset` enthält den Datensatz, der zum Trainieren des Modells verwendet wurde.

Die einzige Klasse ist
0 = drone

## Externer Testdatensatz

Der Datensatz `birds_helicopters.v4i.yolov11` wird ausschließlich für die Auswertung verwendet.

Die Klassen wurden vor der Evaluierung angepasst.
0 = drone
1 = bird
2 = helicopter
3 = plane

Mit diesem Datensatz wird untersucht, ob das Modell ähnliche Objekte als Drohne erkennt.

## Kombinierter Testdatensatz

Mit dem Skript:

create_combined_dataset.py

wird automatisch ein neuer Testdatensatz erstellt.

Dieser enthält

- alle Testbilder des Drohnendatensatzes
- alle Testbilder des externen Datensatzes

Für die Label werden jedoch ausschließlich Drohnen übernommen. Alle anderen Bilder besitzen eine leere Labeldatei.

Dadurch besteht der kombinierte Datensatz nur aus einer Klasse:

0 = drone

Der kombinierte Datensatz wird ausschließlich für die offizielle YOLO-Evaluierung verwendet.

# Training

Das Modell wird mit Ultralytics YOLO11 trainiert.

Beispiel:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="drone_dataset/data.yaml",
    epochs=50,
    imgsz=640
)
```

Nach dem Training befindet sich das beste Modell im Ordner

runs/detect/.../weights/best.pt

# Evaluierung

Für die Evaluierung werden zwei verschiedene Verfahren verwendet.

## 1. Eigene Auswertung

Das Skript:

evaluate_model.py

wertet den externen Datensatz aus.

Dabei werden unter anderem folgende Fragen beantwortet:

- Wie viele Drohnen wurden erkannt?
- Wie viele Vögel wurden als Drohne erkannt?
- Wie viele Hubschrauber wurden als Drohne erkannt?
- Wie viele Flugzeuge wurden als Drohne erkannt?
- Mit welcher Confidence wurden die Objekte erkannt?

Zusätzlich werden

- Diagramme erstellt
- CSV-Dateien gespeichert
- erkannte Bilder mit Bounding Box abgespeichert

Dadurch lassen sich mögliche Fehlklassifikationen einfach analysieren.

## 2. YOLO-Evaluierung

Zusätzlich wird das Modell mit

```python
model.val(...)
```

auf dem kombinierten Testdatensatz ausgewertet.

Dabei werden automatisch die Standardmetriken von YOLO berechnet.

Dazu gehören unter anderem

- Precision
- Recall
- mAP@50
- mAP@50-95
- Precision-Recall-Kurve
- F1-Kurve
- Confusion Matrix

# Ausgabe

Nach der Evaluierung wird automatisch der Ordner:

evaluation_results/

erstellt.

Dieser enthält unter anderem

evaluation_results
│
├── image_results.csv
├── class_statistics.csv
├── yolo_statistics.csv
│
├── detected_as_drone/
│
├── plots/
│
└── yolo_evaluation/

### image_results.csv

Enthält die Ergebnisse jedes einzelnen Bildes.

Beispielsweise

- Bildname
- tatsächliche Klasse
- ob eine Drohne erkannt wurde
- Anzahl der Erkennungen
- maximale Confidence

### class_statistics.csv

Enthält die zusammengefassten Ergebnisse pro Klasse.

Zum Beispiel

- Anzahl der Bilder
- Anzahl der als Drohne erkannten Bilder
- Erkennungsrate
- durchschnittliche Confidence

### plots

In diesem Ordner werden automatisch verschiedene Diagramme gespeichert.

Unter anderem

- Erkennungsrate pro Klasse
- Anzahl erkannter Bilder
- durchschnittliche Confidence
- YOLO-Metriken

### yolo_evaluation

Enthält die automatisch von YOLO erzeugten Auswertungen wie

- Confusion Matrix
- Precision-Recall-Kurve
- F1-Kurve
- weitere Diagramme

# Verwendete Bibliotheken

- Python 3
- Ultralytics YOLO11
- Pandas
- Matplotlib

# Ziel des Projekts

Ziel dieses Projekts war die Entwicklung eines Modells zur Erkennung von Drohnen. Neben der eigentlichen Erkennungsleistung wurde außerdem untersucht, wie häufig das Modell ähnliche Objekte wie Vögel, Hubschrauber oder Flugzeuge fälschlicherweise als Drohne erkennt. Dadurch lässt sich die Zuverlässigkeit des Modells besser einschätzen und mögliche Schwächen des Modells können analysiert werden.

### Referenzen
