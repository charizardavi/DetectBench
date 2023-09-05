import os
import subprocess
import psutil
import time
import matplotlib.pyplot as plt
import threading

from yolov5.yolov5 import Yolov5Model
from yolov8.yolov8 import Yolov8Model

def plot_metrics(metric_values, ylabel, title, filename, color='b'):
        model_names = ["YoloV5", "YoloV8", "EffecientDet-D1", "SSD_640px", "CenterNet_ResNet50", "SSD_320px"]
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

def graph_results():
    
    fps_values = [108.4707631056947, 77.07941985904027, 13.07, 24.67, 22.13, 27]
    map50_values = [0.83, 0.8106877673662868, 0.561, 0.575, 0.666, 0.042]
    map50_95_values = [0.537, 0.5292285064885559, 0.207, 0.292, 0.371, 0.013]
    f1_values = [0.796910569105691, 0.7764258101012973, 0.391, 0.536595, 0.59, 0.0675443156]

    plot_metrics(fps_values, 'FPS', 'FPS for each model', 'fps_performance.png', 'b')
    plot_metrics(map50_values, 'mAP50', 'mAP50 for each model', 'map50_performance.png', 'g')
    plot_metrics(map50_95_values, 'mAP50-95', 'mAP50-95 for each model', 'map50_95_performance.png', 'r')
    plot_metrics(f1_values, 'F1 Score', 'F1 Score for each model', 'f1_performance.png', 'm')


mainPID = os.getpid()
mainMonitor = psutil.Process(mainPID)
exit_signal = False

def getYoloV5_FPS():
    global exit_signal
    yolo = Yolov5Model() 
    yolo.benchmark_fps("people_vid.mp4", os.getcwd()+"/yolov5/yolov5repo/runs/train/exp/weights/best.pt", "yolov5_output.mp4")
    exit_signal = True

def getYoloV5Usage():
    global exit_signal
    cpu_percent_list = []
    ram_usage_list = []
    num_cores = psutil.cpu_count()
    while exit_signal == False:
        cpu_percent = (mainMonitor.cpu_percent()/num_cores)

        memory_info = mainMonitor.memory_info()
        ram_usage = memory_info.rss

        ram_usage_readable = psutil._common.bytes2human(ram_usage)

        cpu_percent_list.append(cpu_percent)
        ram_usage_list.append(ram_usage)
        
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram_usage_readable}")
        time.sleep(1)

    avg_cpu_percent = sum(cpu_percent_list) / len(cpu_percent_list)
    avg_ram_usage = sum(ram_usage_list) / len(ram_usage_list)

    avg_ram_usage_readable = psutil._common.bytes2human(avg_ram_usage)

    print(f"Average CPU Usage: {avg_cpu_percent}%")
    print(f"Average RAM Usage: {avg_ram_usage_readable}")


t1 = threading.Thread(target=getYoloV5_FPS)
t2 = threading.Thread(target=getYoloV5Usage)

t1.start()
t2.start()

t1.join()
t2.join()





exit_signal = False

def getYoloV8_FPS():
    global exit_signal
    yolo = Yolov8Model() 
    yolo.benchmark_fps("people_vid.mp4", "yolov8_output.mp4", os.getcwd()+"/yolov8/runs/detect/train/weights/best.pt")
    exit_signal = True

def getYoloV8Usage():
    global exit_signal
    cpu_percent_list = []
    ram_usage_list = []
    num_cores = psutil.cpu_count()
    while exit_signal == False:
        cpu_percent = (mainMonitor.cpu_percent()/num_cores)

        memory_info = mainMonitor.memory_info()
        ram_usage = memory_info.rss

        ram_usage_readable = psutil._common.bytes2human(ram_usage)

        cpu_percent_list.append(cpu_percent)
        ram_usage_list.append(ram_usage)
        
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram_usage_readable}")
        time.sleep(1)

    avg_cpu_percent = sum(cpu_percent_list) / len(cpu_percent_list)
    avg_ram_usage = sum(ram_usage_list) / len(ram_usage_list)

    avg_ram_usage_readable = psutil._common.bytes2human(avg_ram_usage)

    print(f"Average CPU Usage: {avg_cpu_percent}%")
    print(f"Average RAM Usage: {avg_ram_usage_readable}")


t1 = threading.Thread(target=getYoloV8_FPS)
t2 = threading.Thread(target=getYoloV8Usage)

t1.start()
t2.start()

t1.join()
t2.join()


yolov5 = Yolov5Model()
yolov8 = Yolov8Model()

yolov5.download_data(os.getcwd()+"/yolov5")
yolov8.download_data(os.getcwd()+"/yolov8/dataset")

yolov5.val(os.getcwd()+"/yolov5")
yolov8.val(os.getcwd()+"/yolov8")

