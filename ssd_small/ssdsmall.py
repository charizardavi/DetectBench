from roboflow import Roboflow
import os
from dotenv import load_dotenv
import re
from git import Repo
import urllib.request
import tarfile
import numpy as np
import tensorflow as tf
import cv2
import time

from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder

# SSD MobileNet V2: 320x320px images

"""
Performance:
FPS: TBD
mAP50: TBD
mAP50-95: TBD
F1: TBD
"""

class SSDSmallModel:
    def download_data(self, path: str = os.getcwd()):
        load_dotenv()
        roboflow_key = os.getenv("ROBOFLOW_KEY")
        rf = Roboflow(api_key=roboflow_key)
        project = rf.workspace("titulacin").project("person-detection-9a6mk")
        dataset = project.version(16).download("tfrecord", path)
        print(dataset.location)


    def install_api(self, path: str = os.getcwd()):
        Repo.clone_from("https://github.com/tensorflow/models.git", path+"/tfmodelrepo")
        os.chdir("tfmodelrepo/research/")
        os.system("protoc object_detection/protos/*.proto --python_out=.")
        os.system("cp object_detection/packages/tf2/setup.py .")
        os.system("python -m pip install .")

    def download_checkpoint(self):
        urllib.request.urlretrieve("http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz", "compressed_checkpoint.tar.gz")
        ckp = tarfile.open("compressed_checkpoint.tar.gz")
        ckp.extractall()
        ckp.close()

    def setup_pipeline(self, path: str = os.getcwd()):
        fine_tune_checkpoint = path+"/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
        train_record_fname = path+"/train/People.tfrecord"
        test_record_fname = path+"/test/People.tfrecord"
        label_map_pbtxt_fname = path+"/train/People_label_map.pbtxt"
        batch_size = 8;
        num_steps = 40000;
        num_classes = 1;

        with open(path + "/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config") as f:
            s = f.read()
        
        with open('model_config.config', 'w') as f:
            s = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

            s = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(batch_size), s)

            s = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(num_steps), s)

            s = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(num_classes), s)

            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
            
            f.write(s)


    def train(self, path: str = os.getcwd()):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.system("python tfmodelrepo/research/object_detection/model_main_tf2.py --model_dir={} --pipeline_config_path={}".format(path+"/results", path+"/model_config.config"))


    def eval(self, path: str = os.getcwd()):
        os.system("python tfmodelrepo/research/object_detection/model_main_tf2.py "
           "--pipeline_config_path={} "
           "--model_dir={} "
           "--checkpoint_dir={} "
           "--alsologtostderr".format(path+"/model_config.config", path, path+"/results"))

    
    def load_model(self, path: str = os.getcwd()):
        model = tf.saved_model.load(path+"/saved_model")
        return model.signatures["serving_default"]

    def predict_on_video(self, video_path, output_path, detect_fn):
        # Define the category index for the 'persona' class
        category_index = {1: {'id': 1, 'name': 'persona'}}

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
        
        times = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Start timer
            start_time = time.time()

            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = detect_fn(input_tensor)

            # Extract detection boxes and scores for visualization
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            labels = detections['detection_classes'][0].numpy().astype(int)

            # Visualize the detection boxes on the frame
            viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes,
                labels,
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.5
            )

            # Stop timer
            end_time = time.time()

            # Append the time taken to the times list
            times.append(end_time - start_time)
            
            # Write the frame to the output video
            out.write(frame)

        avg_fps = len(times) / sum(times)
        print(f'FPS: {avg_fps}')
        
        cap.release()
        out.release()

        return avg_fps


    def export_saved_model(self, pipeline_config_path = os.getcwd()+"/model_config.config", trained_checkpoint_dir = os.getcwd()+"/results", output_directory = os.getcwd()):
        export_command = f"python tfmodelrepo/research/object_detection/exporter_main_v2.py --pipeline_config_path {pipeline_config_path} --trained_checkpoint_dir {trained_checkpoint_dir} --output_directory {output_directory}"
        os.system(export_command)

    def getselfpath(self):
        return os.getcwd()
    

    def getpath(self):
        return os.getcwd()+"/effecientdet"


if __name__ == "__main__":
    ssdnet = SSDSmallModel()

    # ssdnet.install_api()
    # ssdnet.download_data()
    # ssdnet.download_checkpoint()
    # ssdnet.setup_pipeline()

    ssdnet.train()
    # ssdnet.export_saved_model()

    # avg_fps = ssdnet.predict_on_video("people_vid.mp4", "output_vid.mp4", ssdnet.load_model())
    # print(f"Processed video at {avg_fps:.2f} FPS")
