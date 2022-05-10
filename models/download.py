# This is for downloading models from nvidia DeepStream SDK

import os
import urllib.request
from os import path
from zipfile import PyZipFile


class model_download(object):
    
    def __init__(self, MODEL_NAME, MODEL_PRE, DIR_PATH, MODEL_URL, FILE_NAME):
        
        self.model_name = MODEL_NAME
        self.model_pre = MODEL_PRE  
        self.dir_path = DIR_PATH
        self.model_url = MODEL_URL
        self.file_name = FILE_NAME
        self.extracted_path = os.path.join(os.path.join(self.dir_path, self.model_name), self.model_pre)
        
    def _download(self):
        
        if not path.isfile(self.model_name):
            print("Downloading pretrained weights for {} model. This may take a while...".format(self.model_name))
            urllib.request.urlretrieve(self.model_url, self.file_name)
            pzf = PyZipFile(self.file_name)
            pzf.extractall(path = self.extracted_path)
            print(f"Dowloaded pre-trained DashCamNet model to: {self.extracted_path}")

    
  