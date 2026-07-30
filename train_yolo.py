from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        plots=True,
        project="runs",
        name="drone_test_50"
    )


if __name__ == "__main__":
    main()

#credits
# @software{yolo11_ultralytics,
#   author = {Glenn Jocher and Jing Qiu},
#   title = {Ultralytics YOLO11},
#   version = {11.0.0},
#   year = {2024},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
# }