import time
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.shufflenetv2_ver2 import ShuffleNetV2, CoordConv
from model.shufflenetv2_ver3 import ShuffleNetV2_ver3
from dataset import FaceLandmarksDataset
from model.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from model.densenet import DenseNet3
# from model.muxnet import factory
from model.skipnet import imagenet_rnn_gate_18
# from model.mobileVit import MobileViT_XXS, MobileViT_XS, MobileViT_S
from model.resnet import ResNet101, ResNet152
from utils import img_show, validate, score, rescale_landmark, check_modelsize,generate_PDB_annot, fix_seed

from loss import PFLDLoss, AdaptiveWingLoss, WingLoss, RectifiedWingLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SIZE = 15

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    score_train, loss_train = 0, 0
    all_land = np.array([]).reshape(0, 68, 2)
    all_pred = np.array([]).reshape(0, 68, 2)
    for img, landmarks in tqdm(train_loader):
        img = img.to(DEVICE)
        landmarks = landmarks.view(landmarks.size(0),-1)
        landmarks = landmarks.to(DEVICE)

        predictions = model(img)
        # print(landmarks.size(), predictions.size())

        optimizer.zero_grad()

        loss = criterion(predictions, landmarks)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        
        pred_numpy = predictions.cpu().detach().numpy()
        pred_numpy = pred_numpy.reshape(pred_numpy.shape[0], -1, 2)

        land_numpy = landmarks.cpu().detach().numpy()
        land_numpy = land_numpy.reshape(pred_numpy.shape[0], -1, 2)


        all_land = np.vstack([all_land, rescale_landmark(land_numpy)])
        all_pred = np.vstack([all_pred, rescale_landmark(pred_numpy)])
        
    assert(all_land.shape == (len(train_loader.dataset), 68, 2))
    assert(all_pred.shape == (len(train_loader.dataset), 68, 2))
    score_train += score(all_land, all_pred)
    loss_train /= len(train_loader)

    return loss_train, score_train

def main(args):
    fix_seed(42)
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # step 2: data
    # 68 landmarks, img size: (384, 384, 3)
    # argumetion
    # print(args.PDB_mode)
    logging.info(f'device: {DEVICE}')
    if args.PDB_mode:
        # if not os.path.exists(os.path.join(args.dataroot, 'PDB_annot.pkl')):
        logging.info('Generate new data')
        generate_PDB_annot(args)
        logging.info('Run with PDB mode')
        
    train_dataset = FaceLandmarksDataset(args.dataroot, mode='train', isPDB=args.PDB_mode,add_feature_channel=args.extra_channel)
    val_dataset = FaceLandmarksDataset(args.val_dataroot, mode='eval', add_feature_channel=args.extra_channel)

    # image, landmarks = train_dataset[0]
    # img_show(image, landmarks, './example.png')

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.train_batchsize, 
                                                shuffle=True, 
                                                num_workers=4)
                                                
    valid_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=args.val_batchsize, 
                                                shuffle=False, 
                                                num_workers=4)
    # Step 2: model, criterion, optimizer, scheduler

    if args.model == "shufflenet_corr":
        model = CoordConv(3, n_class=136,extra_channel=args.extra_channel).to(DEVICE)
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
    elif args.model == "skipnet":
        model = imagenet_rnn_gate_18(device=DEVICE)
    elif args.model == "mobilevit":
        model = MobileViT_XS().to(DEVICE)
    elif args.model == "resnet101":
        model = ResNet101().to(DEVICE)
    elif args.model == "resnet152":
        model = ResNet152().to(DEVICE)

    if not args.train_from_start:
        logging.info(f"Continue training from epoch: {args.start_epoch}")
        model.load_state_dict(torch.load(args.model_path))
    
    model_size = check_modelsize(model)
    if model_size > BASE_SIZE:
        logging.info("Model size too large!!")
        return -1

    if args.loss_mode == "mse":
        criterion = nn.MSELoss()
    elif args.loss_mode == "wing":
        criterion = WingLoss(omega=args.omega, epsilon=args.eps)
    elif args.loss_mode == "adawing":
        criterion = AdaptiveWingLoss()
    elif args.loss_mode == "rec_wing":
        criterion = RectifiedWingLoss()


    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.base_lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.base_lr,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    min_loss = 100

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss, train_score = train(train_loader, model, criterion, optimizer, epoch)

        val_loss, val_score = validate(valid_loader, model, criterion)

        scheduler.step(val_loss)
        # writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {
            'val loss': val_loss,
            'train loss': train_loss
        }, epoch)

        logging.info(f'Epoch[{epoch}/{args.end_epoch}] train loss: {train_loss} / train score: {train_score}')
        logging.info(f'Epoch[{epoch}/{args.end_epoch}] val loss: {val_loss} / val score: {val_score}')

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), args.model_path)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=30, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument('--model_path',
                        default='./checkpoint/model.pkl',
                        type=str,
                        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='cv_data/data/synthetics_train/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='cv_data/data/aflw_val/',
                        type=str,
                        metavar='PATH')

    parser.add_argument('--train_batchsize', default=16, type=int)
    parser.add_argument('--val_batchsize', default=16, type=int)
    parser.add_argument('--PDB_mode', action='store_true')
    parser.add_argument('--loss_mode', default="wing", type=str)
    parser.add_argument('--model', default="shufflenet_corr", type=str)
    parser.add_argument('--train_from_start', action='store_false')
    parser.add_argument('--extra_channel', action='store_true')
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--omega', default=14, type=int)
    parser.add_argument('--eps', default=2, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
