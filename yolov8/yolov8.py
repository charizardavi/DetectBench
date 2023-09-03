from roboflow import Roboflow
from ultralytics import YOLO
import ultralytics
import os
from dotenv import load_dotenv
import cv2
import time
from contextlib import redirect_stdout
import logging

"""
Performance:
FPS: 77.07941985904027
mAP50: 0.8106877673662868
mAP50-95: 0.5292285064885559
F1: 0.7764258101012973
"""

class Yolov8Model:
    def download_data(self, path: str):
        load_dotenv()
        roboflow_key = os.getenv("ROBOFLOW_KEY")
        rf = Roboflow(api_key=roboflow_key)
        project = rf.workspace("titulacin").project("person-detection-9a6mk")
        dataset = project.version(16).download("yolov8", path)
        rPath = dataset.location+"/data.yaml"
        
        Yolov8Model.replace_path(rPath, "test", "test: test/images")
        Yolov8Model.replace_path(rPath, "train", "train: train/images")
        Yolov8Model.replace_path(rPath, "val", "val: valid/images")
        
        f = open("dataset_loc.txt", "w+")
        f.write(dataset.location)
        f.close()

    def train(self):
        dataset_loc = ""
        with open('dataset_loc.txt') as f:
            dataset_loc = f.readline()
        
        model = YOLO('yolov8n.pt')
        model.train(data=dataset_loc+"/data.yaml", epochs=100, imgsz=640)

    def replace_path(file_path, search_str, replace_str):
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the line with the search_str and replace the entire line
        for i, line in enumerate(lines):
            if search_str in line:
                lines[i] = replace_str + "\n"  # Add a newline to replace the entire line

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

    def mAP50(self, yolov8path:str):
        model = YOLO(yolov8path+'/runs/detect/train/weights/best.pt')
        metrics = model.val(yolov8path+"/dataset/data.yaml") 
        return metrics.box.map50
    
    def mAP50_95(self, yolov8path:str):
        model = YOLO(yolov8path+'/runs/detect/train/weights/best.pt')
        metrics = model.val(yolov8path+"/dataset/data.yaml")
        return metrics.box.map
    
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category



    def val(self, yolov8path:str):
        model = YOLO(yolov8path+'/runs/detect/train/weights/best.pt')
        metrics = model.val(yolov8path+"/dataset/data.yaml") 
        P = metrics.results_dict["metrics/precision(B)"]
        R = metrics.results_dict["metrics/recall(B)"]
        if P + R == 0:
            return 0
        return 2 * (P * R) / (P + R)

    

    def benchmark_fps(self, video_path: str, output_path: str, model_path: str):
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

        # Initialize YOLO model
        model = YOLO(model_path)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # with open(os.devnull, 'w') as fnull:
            #     with redirect_stdout(fnull):
            results = model(frame)
            
            # Loop through all results (for each frame)
            for result in results:
                if result.boxes is not None:  # Check if there are any detections
                    boxes = result.boxes.xyxy  # Extract bounding boxes in (x1, y1, x2, y2) format
                    class_labels = result.boxes.cls  # Extract class labels
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        label = int(class_labels[i])
                        class_name = result.names[label]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps}")
        return fps

    def getpath(self):
        return os.getcwd()+"/yolov8"
    
    def getdatapath(self):
        return os.getcwd()+"/yolov8/dataset"

    def getselfpath(self):
        return os.getcwd()
    
    def getselfdatapath(self):
        return os.getcwd()+"/dataset"

if __name__ == "__main__":
    yolo = Yolov8Model()
    # video_path = 'people_vid.mp4'  # Replace with your video path
    # model_weights = 'runs/detect/train/weights/best.pt'  # Replace with your model weights
    # yolo.benchmark_fps(video_path, model_weights)
    # print(yolo.mAP50(yolo.getselfpath()))
    # print(yolo.mAP50_95(yolo.getselfpath()))
    # print(yolo.val(yolo.getselfpath()))
    # yolo.download_data(yolo.getselfdatapath())
    # yolo.val(yolo.getselfdatapath())
    yolo.train()
