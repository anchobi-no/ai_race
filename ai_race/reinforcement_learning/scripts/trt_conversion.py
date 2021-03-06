#!/usr/bin/env python
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np
import time
from PIL import Image as IMG
from torch2trt import TRTModule
import cv2
from cv_bridge import CvBridge

DISCRETIZATION = 2

from networks.simplenet import DQN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_inference():
    global device
    
    model = DQN(120, 320, DISCRETIZATION)
    model.eval()
    
    model.load_state_dict(torch.load(args.pretrained_model))
    model = model.cuda()
    x = torch.ones((1, 3, 120, 320)).cuda()

    from torch2trt import torch2trt
    model_trt = torch2trt(model, [x], max_batch_size=100)
    torch.save(model_trt.state_dict(), args.trt_model)


def parse_args():
    # Set arguments.
    arg_parser = argparse.ArgumentParser(description="Autonomous with inference")
	
    arg_parser.add_argument("--pretrained_model", type=str)
    arg_parser.add_argument("--trt_model", type=str, default='~/catkin_ws/src/ai_race/ai_race/reinforcement_learning/trt_model/dqn_20210108_n2_trt.pth' )

    args = arg_parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("process start...")

    init_inference()

    print("finished successfully.")
    print("model_path: " + args.trt_model)
    os._exit(0)
