import os
from download import model_download



FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

#MODEL_NAME = "dashcamnet"
#MODEL_PRE = "tlt_dashcamnet_pruned_v2.0"
#MODEL_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0/zip"
#FILE_NAME = "/tmp/dashcamnet_pruned_v1.0.zip"

MODEL_NAME = "peoplenet"
MODEL_PRE = "tlt_peoplenet_pruned_v2.0"
MODEL_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/tlt_peoplenet/versions/pruned_v2.0/zip"
FILE_NAME = "/tmp/tlt_peoplenet_pruned_v2.0.zip"



if __name__ == "__main__":

    model_download(MODEL_NAME, MODEL_PRE, DIR_PATH, MODEL_URL, FILE_NAME)._download()
