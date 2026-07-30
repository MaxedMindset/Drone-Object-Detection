# Datensätze

Die Datensätze werden wegen Größe und Nutzungsbedingungen nicht im Repository gespeichert.

## Trainingsdatensatz

SSHIKAMARU, **Drone Object Detection**, Kaggle:

https://www.kaggle.com/datasets/sshikamaru/drone-yolo-detection

Die Bilder und Label werden zunächst gemeinsam in `Database1/` abgelegt. Danach:

```bash
python scripts/prepare_dataset.py
```

Erwartete Ausgabe:

```text
drone_dataset/
├── data.yaml
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
└── labels/test
```

## Externer Testdatensatz

Erwartete Struktur:

```text
birds_helicopters.v4i.yolov11/
└── test/
    ├── images/
    └── labels/
```

Klassenbelegung vor der Auswertung:

```text
0 = drone
1 = bird
2 = helicopter
3 = plane
```

Genauer Roboflow-Link:

```text
HIER_DEN_GENAUEN_LINK_EINTRAGEN
```
