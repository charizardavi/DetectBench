from roboflow import Roboflow
import os
from dotenv import load_dotenv
import torch
from git import Repo



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
        os.system("python yolov5repo/val.py  --data yolov5repo/data.yaml --weights yolov5repo/runs/train/exp3/weights/best.pt")
        




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
    yolo.val()
    # yolo.download_data(yolo.getselfpath())
    # yolo.train()
    # yolo.predict()