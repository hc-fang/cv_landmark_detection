import os
import cv2
import pickle

import torch
import numpy as np
from glob import glob

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import imutils
import cv2 
from math import cos, sin, radians

def Add_feature_channel(img):
    H,W=img.shape[0],img.shape[1]
    canvas=np.zeros([H,W])
    canvas=canvas.astype(np.uint8)
    Widx_crop=np.round(W*0.2).astype(np.int)
    Hidx_crop=np.round(H*0.05).astype(np.int)
    img_crop=img[Hidx_crop:H-Hidx_crop,Widx_crop:W-Widx_crop]
    orb = cv2.ORB_create(scoreType=1,nfeatures=300)
    kp = orb.detect(img_crop,None)
    kp_point=[i.pt for i in kp]
    kp_point=np.array(np.round(kp_point)).astype(int)
    
    if kp_point.shape[0]!=0:
      
       canvas[kp_point[:,1]+Hidx_crop,kp_point[:,0]+Widx_crop]=255
    canvas= cv2.GaussianBlur(canvas,(5,5),0).reshape(H,W,1)

    return canvas

class Transforms():
    def __init__(self, mode):
        self.mode = mode
        
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def randomFlip(self, image, landmarks, p=0.5):
        if torch.rand(1) < p:
            landmarks[0]=np.array(image).shape[1]-landmarks[0]
            return transforms.RandomHorizontalFlip(p=1)(image),landmarks 
        return image,landmarks

    def randomPosterize(self, image, landmarks, p=0.5,bits=5):
        return transforms.RandomPosterize(bits, p=0.5)(image),landmarks 

    def gaussianBlur(self, image, landmarks,kernel_size=(5, 11),sigma=(5, 10)):
        return  transforms.GaussianBlur(kernel_size,sigma)(image),landmarks 
      
           
    def norm_landmark(self, image, landmarks):
        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks)
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def channel_shuffle(self, img, annotation):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
        return img, annotation

    def random_noise(self, img, annotation, limit=[0, 0.2], p=0.5):
        if random.random() < p:
            H, W = img.shape[:2]
            noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

            img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img, annotation

    def __call__(self, image, landmarks=None):
        image = Image.fromarray(image)
        if self.mode == 'test':
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])
            return image
        elif self.mode == 'eval':
            image, landmarks = self.norm_landmark(image, landmarks)
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])
            return image, landmarks
        else:
            image, landmarks = self.norm_landmark(image, landmarks)
            # image, landmarks = self.resize(image, landmarks, (224, 224))

            # more data argumentation
            # image, landmarks = self.gaussianBlur(image, landmarks)   
            # image, landmarks = self.randomPosterize(image, landmarks)   
            image, landmarks = self.color_jitter(image, landmarks)

            image, landmarks = self.rotate(image, landmarks, angle=10)

            # image, landmarks = self.random_noise(image, landmarks)         

            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])
            return image, landmarks


class FaceLandmarksDataset():
    def __init__(self, root, mode, isPDB=False,add_feature_channel=False):
        self.add_feature_channel=add_feature_channel
        self.root = root
        self.mode = mode
        self.transform = Transforms(mode=self.mode)
        if self.mode == 'test':
            self.image_filenames = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
            print(f"test example: {len(self.image_filenames)}")
        else:
            if isPDB and self.mode == 'train':
                with open(os.path.join(root, 'PDB_annot.pkl'), 'rb') as f:
                    annot = pickle.load(f)
                    self.image_filenames, self.landmarks = annot
            else:
                with open(os.path.join(root, 'annot.pkl'), 'rb') as f:
                    annot = pickle.load(f)
                    self.image_filenames, self.landmarks = annot

            self.landmarks = np.array(self.landmarks).astype('float32')

            print('image shape: ', self.landmarks.shape)
            print('landmarks shape: ', self.landmarks.shape)
            
            assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, self.image_filenames[index]))
        if self.add_feature_channel:
            image_extra=Add_feature_channel(image)
            image=np.append(image,image_extra, axis=2)
        if self.mode == 'test':
            image = self.transform(image)
            return self.image_filenames[index], image
        else:
            landmarks = self.landmarks[index]
            image, landmarks = self.transform(image, landmarks)
            # normalize and make its mean to 0
            landmarks = landmarks - 0.5
           
            return image, landmarks