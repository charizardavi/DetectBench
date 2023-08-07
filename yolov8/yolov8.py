from roboflow import Roboflow
from ultralytics import YOLO
import os
from dotenv import load_dotenv



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

    def val(self, yolov8path:str):
        print(yolov8path)
        model = YOLO(yolov8path+'/runs/detect/train14/weights/best.pt')
        metrics = model.val(yolov8path+"/dataset/data.yaml")  # no arguments needed, dataset and settings remembered
        return metrics.box.map50
        # metrics.box.map    # map50-95
        # metrics.box.map50  # map50
        # metrics.box.map75  # map75
        # metrics.box.maps   # a list contains map50-95 of each category

    def getpath(self):
        return os.getcwd()+"/yolov8"
    
    def getdatapath(self):
        return os.getcwd()+"/yolov8/dataset"

    def getselfpath(self):
        return os.getcwd()
    
    def getselfdatapath(self):
        return os.getcwd()+"/dataset"

if __name__ == "__main__":
    print(os.getcwd())
    # yolo.download_data(yolo.getselfdatapath())
    # yolo.val(yolo.getselfdatapath())
    # yolo.train()
