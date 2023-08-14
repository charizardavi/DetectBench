from roboflow import Roboflow
import os
from dotenv import load_dotenv
from roboflow import Roboflow
import re
from git import Repo
import urllib.request
import tarfile

# specifically uses effecientdet d1: 640x640px images
# train script: tfmodelrepo/research/object_detection/model_main_tf2.py

class CenternetModel:
    def download_data(self, path: str):
        # Repo.clone_from("https://github.com/tensorflow/models.git", path+"/tfmodelrepo")
        # os.chdir("tfmodelrepo/research/")
        # os.system("protoc object_detection/protos/*.proto --python_out=.")
        # os.system("cp object_detection/packages/tf2/setup.py .")
        # os.system("python -m pip install .")

        # urllib.request.urlretrieve("https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/centernet_hourglass104_512x512_coco17_tpu-8.config", "model_config_template.config")
        # urllib.request.urlretrieve("http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz", "compressed_checkpoint.tar.gz")
        # ckp = tarfile.open("compressed_checkpoint.tar.gz")
        # ckp.extractall()
        # ckp.close()
        
        # extracted tar folder: centernet_hg104_512x512_coco17_tpu-8
        
        load_dotenv()
        roboflow_key = os.getenv("ROBOFLOW_KEY")
        rf = Roboflow(api_key=roboflow_key)
        project = rf.workspace("titulacin").project("person-detection-9a6mk")
        dataset = project.version(16).download("tfrecord", path)
        
        fine_tune_checkpoint = os.getcwd()+"/centernet_hg104_512x512_coco17_tpu-8/checkpoint/ckpt-0"
        train_record_fname = os.getcwd()+"/train/People.tfrecord"
        test_record_fname = os.getcwd()+"/test/People.tfrecord"
        label_map_pbtxt_fname = os.getcwd()+"/train/People_label_map.pbtxt"
        batch_size = 16;
        num_steps = 40000;
        num_classes = 1;

        with open("model_config_template.config") as f:
            s = f.read()
        
        with open('model_config.config', 'w') as f:
            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

            # tfrecord files train and test.
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(num_steps), s)

            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(num_classes), s)

            #fine-tune checkpoint type
            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

            f.write(s)
    def train(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.system("python tfmodelrepo/research/object_detection/model_main_tf2.py --model_dir={} --pipeline_config_path={}".format(os.getcwd()+"/results", os.getcwd()+"/model_config.config"))



    def getselfpath(self):
        return os.getcwd()


if __name__ == "__main__":
    cenet = CenternetModel()
    # cenet.download_data(cenet.getselfpath())
    cenet.train()