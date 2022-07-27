import time
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import FaceLandmarksDataset
from model.resnet import ResNet
from model.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
# from model.muxnet import factory
from utils import img_show, validate, score, rescale_landmark, check_modelsize, fix_seed
from model.mobileVit import MobileViT_XXS, MobileViT_XS, MobileViT_S
from model.shufflenetv2_ver2 import ShuffleNetV2, CoordConv
from model.shufflenetv2_ver3 import ShuffleNetV2_ver3

from loss import PFLDLoss, AdaptiveWingLoss, WingLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SIZE = 15



def main(args):
    fix_seed(42)
    val_dataset = FaceLandmarksDataset(args.val_dataroot, mode='eval', add_feature_channel=args.extra_channel)

    # image, landmarks = train_dataset[0]
    # img_show(image, landmarks, './example.png')


    valid_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=args.val_batchsize, 
                                                shuffle=False, 
                                                num_workers=0)

    if args.model == "shufflenet_corr":
        model = CoordConv(3, n_class=136, extra_channel=args.extra_channel).to(DEVICE)
    elif args.model =="shufflenetv2_ver3":
        model = ShuffleNetV2_ver3(3, n_class=136,extra_channel=args.extra_channel).to(DEVICE)
    elif args.model == "resnet":
        model = ResNet().to(DEVICE)
    elif args.model == "shufflenet":
        model = ShuffleNetV2(n_class=136).to(DEVICE)
    elif args.model == "densenet":
        model = DenseNet3(num_classes=136, depth=30).to(DEVICE)
    # elif args.model == "muxnet":
    #     model = factory("muxnet_m", pretrained=False, num_classes=136).to(DEVICE)

    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    model_size = check_modelsize(model)
    if model_size > BASE_SIZE:
        print("Model size too large!!")
        return -1
        

    if args.loss_mode == "mse":
        criterion = nn.MSELoss()
    elif args.loss_mode == "wing":
        criterion = WingLoss()
    elif args.loss_mode == "adawing":
        criterion = AdaptiveWingLoss()

    val_loss = validate(valid_loader, model, criterion)
    


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss_mode', default="wing", type=str)
    parser.add_argument('--model',
                        default='shufflenet_corr',
                        type=str)

    parser.add_argument('--model_path',
                        default='./checkpoint/model.pkl',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--val_dataroot',
                        default='cv_data/data/aflw_val/',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--val_batchsize', default=1, type=int)
    parser.add_argument('--extra_channel', action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
