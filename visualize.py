import time
import random
import argparse
import logging
import numpy as np

import torch
from utils import visualize
from torch.utils.data import DataLoader

from dataset import FaceLandmarksDataset
from model.resnet import ResNet
from model.mobilenetv3 import mobilenetv3_small, mobilenetv3_large

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    val_dataset = FaceLandmarksDataset(args.val_dataroot)

    valid_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=args.val_batchsize, 
                                                shuffle=True, 
                                                num_workers=4)

    model = mobilenetv3_small()
    model.to(DEVICE)
    model.load_state_dict(torch.load(args.model_path)) 
    visualize(valid_loader, model, args.output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default='./checkpoint/model.pkl',
                        type=str,
                        metavar='PATH')
    # --dataset         
    parser.add_argument('--output_path',
                        default='./val_image.png',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--val_dataroot',
                        default='/tmp2/b09902120/cv_data/data/aflw_val/',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)