from roboflow import Roboflow
import os
from dotenv import load_dotenv
import torch
from git import Repo
import cv2
import time
import subprocess
import re
"""
Performance:
FPS: 108.4707631056947
F1: 0.796910569105691
mAP50: .83
mAP50-95: 0.537
"""
class Yolov5Model:
    def download_data(self, path:str):
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
    
    def val(self):
        try:
            val_output = subprocess.check_output([
                'python', 
                'yolov5repo/val.py',
                '--data', 'yolov5repo/data.yaml',
                '--weights', 'yolov5repo/runs/train/exp3/weights/best.pt'
            ], stderr=subprocess.STDOUT).decode('utf-8')

            print("Captured output: ")
            print(val_output)

            # Extracting mAP and Precision, Recall for F1 calculation
            map_value = re.search(r"mAP50-95\s+=\s+(\d+\.\d+)", val_output)
            p_value = re.search(r"P\s+=\s+(\d+\.\d+)", val_output)
            r_value = re.search(r"R\s+=\s+(\d+\.\d+)", val_output)

            if map_value and p_value and r_value:
                map_value = float(map_value.group(1))
                p_value = float(p_value.group(1))
                r_value = float(r_value.group(1))

                # Calculating F1 Score
                f1_value = 2 * (p_value * r_value) / (p_value + r_value)

                print(f"Validation Metrics: mAP: {map_value}, F1: {f1_value}")

            else:
                print("Could not find mAP and F1 values.")

        except subprocess.CalledProcessError as e:
            print(f"Validation script failed with error: {e.output.decode('utf-8')}")
        
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
    
    def getF1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    
    
    # def predict(self):
    #     model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    #     img = "https://ultralytics.com/images/zidane.jpg"
    #     results = model(img)
    #     results.print()

if __name__ == "__main__":
    yolo = Yolov5Model()
    video_path = 'people_vid.mp4'  # Replace with your video path
    model_weights = 'yolov5repo/runs/train/exp3/weights/best.pt'  # Replace with your model weights
    yolo.benchmark_fps(video_path, model_weights)
    # yolo.val()
    # print(yolo.getF1(.845, .754))
    # yolo.download_data(yolo.getselfpath())
    # yolo.train()
    # yolo.predict()