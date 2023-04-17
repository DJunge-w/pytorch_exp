import os
import argparse
import json
import collections
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns, sleep
import random
from typing import List
import time
import datetime;
import sys
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F

def read_img(img_path: str):
    img: np.ndarray = cv2.cvtColor(
        cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torchvision.transforms.ToTensor()(img)

class SomeModel():
    def __init__(self, device, model_size) -> None:
        self.layers = []
        self.model_size = model_size
        self.layer = torch.rand((1500, 1500)).to(device)
        # for i in range(50):
            # self.layers.append(torch.rand((2000, 2000)).cuda())
        self.input = torch.rand((20, 1500)).to(device)
    
    def evaluate(self):
        res = self.input
        for j in range(self.model_size):
            res = torch.matmul(res, self.layer)
        return res
        
        

class SchedulerTester():
    def __init__(self, config, device) -> None:
        self.config = config
        model_name = self.config['model_name']
        model_weight = self.config['model_weight']
        if (getattr(torchvision.models.segmentation, model_weight, False)):
            # a model from torchvision.models.segmentation
            self.weights= getattr(torchvision.models.segmentation, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models.segmentation, model_name)(weights=self.weights).eval().to(device)
        elif(getattr(torchvision.models.detection, model_weight, False)):
            # a model from torchvision.models.detection
            self.weights= getattr(torchvision.models.detection, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models.detection, model_name)(weights=self.weights).eval().to(device)
        else:
            # a model from torchvision.models or terminated with an excepton
            self.weights= getattr(torchvision.models, model_weight).DEFAULT
            self.model: torch.nn.Module = getattr(torchvision.models, model_name)(weights=self.weights).eval().to(device)

        #No resize FasterRCNN_ResNet50 720
        self.model_preprocess = self.weights.transforms()
        img_path = self.config['input_file_path'] 
        self.img: torch.Tensor =\
            self.model_preprocess(read_img(img_path)).unsqueeze(0).to(device)

    def infer(self):
        st: int = perf_counter_ns()
        # print(f"starting inference, idx: {self.idx} at {int(time.time() * 1000000000) // 1000 % 100000000 / 1000.0}")
        res: torch.Tensor = None
        res = self.model(self.img) 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"JCT {clock_gettime_ns(CLOCK_REALTIME)}"
                     f" {(perf_counter_ns() - st) / 1000000}\n")
        return res

    def run(self):
        sleep_dur = self.config['sleep_time']
        while True:
            if torch.cuda.is_available():
                #for NSight region
                torch.cuda.nvtx.range_push("regionTest")
            self.infer()
            sys.stdout.flush()
            if torch.cuda.is_available():
                #for NSight region
                torch.cuda.nvtx.range_pop()
            sleep(sleep_dur)


def main():  
    #Get input model configuration file path
    parser = argparse.ArgumentParser(description="Run a model's inference job")
    parser.add_argument('filename', help="Specifies the path to the model JSON file")
    parser.add_argument('deviceid', help="Specifies the gpu to run")
    args = parser.parse_args()
    filename = args.filename
    device_id = args.deviceid

    print("Device", int(device_id))
    # set the directory for downloading models
    torch.hub.set_dir("../torch_cache/")

    # set the cuda device to use
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(int(device_id))
    else:
        device = torch.device("cpu")
    
    try:
        print(f"run_model.py: parsing file: {filename}")
        with open(filename, 'r') as file_input:
            data = json.load(file_input, object_pairs_hook=collections.OrderedDict)
            print(f"Model: {data['model_name']}")
            print("Fields:")
            print(json.dumps(data, indent=4))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print(f"Invalid input file.", file=sys.stderr)
        sys.exit(1)

    # tester: SchedulerTester = SchedulerTester(0)    
    tester: SchedulerTester = SchedulerTester(data, device)
    sleep(1)
    tester.run()




if __name__ == "__main__":
    main()
