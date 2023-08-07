from yolov8.yolov8 import Yolov8Model
from yolov5.yolov5 import Yolov5Model
import os
import subprocess



if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # yolov5_path = os.path.join(current_directory, 'yolov5')
    # subprocess.run(['python', 'yolov5.py'], cwd=yolov5_path)

    efdet_path = os.path.join(current_directory, 'effecientdet')
    subprocess.run(['python', 'effecientdet.py'], cwd=efdet_path)


    # yolov8 = Yolov8Model();
    # print(yolov8.val(yolov8.getpath()))
    # yolov5 = Yolov5Model()
    # yolov5.download_data()
    # yolov5.train()