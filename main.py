"""
from yolov8.yolov8 import Yolov8Model
from yolov5.yolov5 import Yolov5Model
import os
import subprocess
"""

import matplotlib.pyplot as plt

model_names = ["YoloV5", "YoloV8", "EffecientDet-D1", "SSD_Mobilenet", "CenterNet_ResNet50"]
fps_values = [108.4707631056947, 77.07941985904027, 13.07, 24.67, 22.13]
map50_values = [0.83, 0.8106877673662868, 0.561, 0.575, 0.666]
map50_95_values = [0.537, 0.5292285064885559, 0.207, 0.292, 0.371]
f1_values = [0.796910569105691, 0.7764258101012973, 0.391, 0.536595, 0.59]

def plot_metrics(metric_values, ylabel, title, filename, color='b'):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, metric_values, color=color)
    plt.xlabel('Model Names')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add labels on bars
    for bar, value in zip(bars, metric_values):
        if value is not None:  # Only label bars with actual values
            plt.text(bar.get_x() + bar.get_width()/2, value, round(value, 3), ha='center', va='bottom')

    plt.savefig(filename)
    plt.close()

plot_metrics(fps_values, 'FPS', 'FPS for each model', 'fps_performance.png', 'b')
plot_metrics(map50_values, 'mAP50', 'mAP50 for each model', 'map50_performance.png', 'g')
plot_metrics(map50_95_values, 'mAP50-95', 'mAP50-95 for each model', 'map50_95_performance.png', 'r')
plot_metrics(f1_values, 'F1 Score', 'F1 Score for each model', 'f1_performance.png', 'm')



"""
    current_directory = os.path.dirname(os.path.abspath(__file__))

    yolov5_path = os.path.join(current_directory, 'yolov5')
    subprocess.run(['python', 'yolov5.py'], cwd=yolov5_path)

    efdet_path = os.path.join(current_directory, 'effecientdet')
    subprocess.run(['python', 'effecientdet.py'], cwd=efdet_path)


    yolov8 = Yolov8Model();
    print(yolov8.val(yolov8.getpath()))
    yolov5 = Yolov5Model()
    yolov5.download_data()
    yolov5.train()
    """