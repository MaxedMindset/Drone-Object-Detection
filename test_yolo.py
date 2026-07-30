from ultralytics import YOLO
import os
import time


def main():

    model = YOLO("runs/detect/runs/drone_test_50/weights/best.pt")

    print(os.path.exists("drone_dataset/images/test"))
    print(os.listdir("drone_dataset/images/test")[:10])

    model.predict(
        source="drone_dataset/images/test",
        conf=0.25,
        save=True,
        show=True
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