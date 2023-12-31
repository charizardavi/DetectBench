from roboflow import Roboflow
import os
from dotenv import load_dotenv
import torch
from git import Repo
import cv2
import time
import subprocess
import re
from contextlib import redirect_stdout

"""
Performance:
FPS: 108.4707631056947
mAP50: .83
mAP50-95: 0.537
F1: 0.796910569105691
"""
class Yolov5Model:
    def download_data(self, path:str = os.getcwd()):
        Repo.clone_from("https://github.com/ultralytics/yolov5.git", path+"/yolov5repo")
        global dataset
        load_dotenv()
        roboflow_key = os.getenv("ROBOFLOW_KEY")
        rf = Roboflow(api_key=roboflow_key)
        project = rf.workspace("titulacin").project("person-detection-9a6mk")
        dataset = project.version(16).download("yolov5", path+"/yolov5repo")

        rPath = dataset.location+"/data.yaml"
        
        Yolov5Model.replace_path(rPath, "test", "test: test/images")
        Yolov5Model.replace_path(rPath, "train", "train: train/images")
        Yolov5Model.replace_path(rPath, "val", "val: valid/images")

        f = open("dataset_loc.txt", "w+")
        f.write(dataset.location)
        f.close()
        
        
        
        
    
    def train(self):
        os.system("python yolov5repo/train.py --img 640 --batch 16 --epochs 100 --data yolov5repo/data.yaml --weights yolov5s.pt --cache")
    
    def val(self, yolov5path = os.getcwd()):
        os.system('python {}/yolov5repo/val.py --data {}/yolov5repo/data.yaml --weights {}/best.pt'.format(yolov5path, yolov5path, yolov5path))
        
    def benchmark_fps(self, video_path, model_weights, output_path='output_video.mp4'):
        # Initialize the model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)

        # Initialize OpenCV video capture
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize variables to hold time samples
        times = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Record the start time
            start_time = time.time()

            # Make the prediction
            results = model(frame)

            # Record the end time
            end_time = time.time()

            # Append the time taken to the times list
            times.append(end_time - start_time)

            # Render the results on the frame
            rendered_frame = results.render()[0]

            # Write the frame to the output video
            out.write(rendered_frame)

            # Show the frame if you wish (Optional)
            # cv2.imshow('Frame', rendered_frame)

            # Press Q to quit (Optional)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Calculate FPS
        avg_fps = len(times) / sum(times)
        print(f'FPS: {avg_fps}')
        return avg_fps



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


    def getpath(self):
        return os.getcwd()+"/yolov5"
    
    def getselfpath(self):
        return os.getcwd()
    
    
    
    # def predict(self):
    #     model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    #     img = "https://ultralytics.com/images/zidane.jpg"
    #     results = model(img)
    #     results.print()

if __name__ == "__main__":
    yolo = Yolov5Model()
    # yolo.download_data(yolo.getselfpath())
    # yolo.train()
    # yolo.val()
    
