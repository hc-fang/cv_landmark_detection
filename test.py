import time
import random
import argparse
import logging
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import FaceLandmarksDataset
from model.resnet import ResNet
from model.shufflenetv2_ver2 import ShuffleNetV2,CoordConv
from model.shufflenetv2_ver3 import ShuffleNetV2_ver3
from model.mobilenetv3 import mobilenetv3_small
from model.mobileVit import MobileViT_XXS, MobileViT_XS, MobileViT_S
from utils import img_show, rescale_landmark

from utils import img_show, rescale_landmark, fix_seed
import subprocess


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def visualize_result(imgs, predictions, names, num=10):
    # Generate image takes some time, so only output the first 10 
    imgs = imgs[:num]
    predictions = predictions[:num]
    names = names[:num]
    for img, prediction, name in tqdm(zip(imgs, predictions, names)):
        prediction = np.array(prediction).reshape(-1, 2)
        result_dir = os.path.join(args.output_dir, "results")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        path = os.path.join(result_dir, name)
        img_show(img, prediction, path)

def main(args):
    fix_seed(42)
    test_dataset = FaceLandmarksDataset(os.path.join(args.data_dir, "aflw_test"), mode='test', add_feature_channel=args.extra_channel)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=False, 
                                                num_workers=4)
                                
    

    if args.model == "shufflenet_corr":
         model = CoordConv(3, n_class=136, extra_channel=args.extra_channel).to(DEVICE)
    elif args.model =="shufflenetv2_ver3":
        model = ShuffleNetV2_ver3(3, n_class=136,extra_channel=args.extra_channel).to(DEVICE)
    elif args.model == "resnet":
        model = ResNet().to(DEVICE)
    elif args.model == "shufflenet":
        model = ShuffleNetV2(n_class=136).to(DEVICE)
    # model = mobilenetv3_small().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    predictions = []
    names = []
    imgs = []
    with torch.no_grad():
        for name, img in tqdm(test_loader):
            img = img.to(DEVICE)
            imgs.extend(img.cpu().detach().numpy())
            pred = model(img)
            pred = pred.cpu().detach().numpy()
            rescale_pred = rescale_landmark(pred)
            predictions.extend(rescale_pred.tolist())
            names.extend(list(name))
        
    
    visualize_result(imgs, predictions, names)
    assert(np.array(predictions).shape == (1790, 136) and len(names) == 1790)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "solution.txt"), "w") as f:
        for name, prediction in zip(names, predictions):
            out_line = f"{name} " + " ".join(map(str, prediction)) + '\n'
            f.write(out_line)
        print(f"testing finish! Results in {args.output_dir}")
        # subprocess.call(["zip", "cv_data/output/solution.zip", "cv_data/output/solution.txt"])
            

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        default='shufflenet_corr',
                        type=str)

    parser.add_argument('--model_path',
                        default='./checkpoint/model.pkl',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--data_dir',
                        default='./cv_data/data',
                        type=str,
                        help="Path to data directory")

    parser.add_argument('--output_dir',
                        default='./output',
                        type=str,
                        help="Path to output directory")

    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--extra_channel', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)