from roboflow import Roboflow
import os
from dotenv import load_dotenv
import re
from git import Repo
import urllib.request
import tarfile

# SSD MobileNet V2 FPNLite: 640x640px images

class MobileNetModel:
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
        urllib.request.urlretrieve("http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz", "compressed_checkpoint.tar.gz")
        ckp = tarfile.open("compressed_checkpoint.tar.gz")
        ckp.extractall()
        ckp.close()

    def setup_pipeline(self, path: str = os.getcwd()):
        fine_tune_checkpoint = path+"/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
        train_record_fname = path+"/train/People.tfrecord"
        test_record_fname = path+"/test/People.tfrecord"
        label_map_pbtxt_fname = path+"/train/People_label_map.pbtxt"
        batch_size = 5;
        num_steps = 40000;
        num_classes = 1;

        with open(path + "/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config") as f:
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
        os.system("python object_detection/model_main_tf2.py "
           "--pipeline_config_path={} "
           "--model_dir={} "
           "--checkpoint_dir={} "
           "--alsologtostderr").format(path+"/model_config.config", path, "TRAINING OUTPUT DIR HERE")


    def getselfpath(self):
        return os.getcwd()
    

    def getpath(self):
        return os.getcwd()+"/effecientdet"


if __name__ == "__main__":
    ssdnet = MobileNetModel()

    ssdnet.install_api()
    ssdnet.download_data(ssdnet.getselfpath())
    ssdnet.setup_pipeline()

    MobileNetModel.train()