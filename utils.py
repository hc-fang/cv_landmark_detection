import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
import pickle
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rescale_landmark(landmarks):
    return (landmarks + 0.5) * 384

# score for multiple landmarks, shaoe: (x, 68, 2)
def score(landmarks, predictions):
    # dis = landmarks - predictions
    # dis = np.sqrt(np.sum(np.power(dis, 2), 2))
    # dis = np.mean(dis, axis=1)
    # dis = np.sum(dis)
    # x = dis / 384
    score = 0
    for i in range(len(landmarks)):
        score += score_each(landmarks[i], predictions[i])

    return score/len(landmarks)

def score_each(landmark, prediction):
    # print(landmark.shape)
    dis = (landmark - prediction)
    dis = np.sqrt(np.sum(np.power(dis, 2), 1))
    dis = np.mean(dis)
    x = dis / 384
    # print(x.shape)
    return x

def img_show(image, landmarks, save_path):
    # unnormalized image
    image = (image + 1) /2
    
    plt.figure(figsize=(10, 10))

    plt.imshow(image.transpose(1, 2, 0))
    
    plt.scatter(landmarks[:,0], landmarks[:,1], s=8)
    plt.savefig(save_path)
    plt.close()

def validate(valid_loader, model, criterion):
    model.eval()
    score_valid, loss_valid = 0, 0

    all_land = np.array([]).reshape(0, 68, 2)
    all_pred = np.array([]).reshape(0, 68, 2)
    with torch.no_grad():
        for img, landmarks in valid_loader:
            img = img.to(DEVICE)
            landmarks = landmarks.view(landmarks.size(0),-1).cuda()
            predictions = model(img)

            # find the loss for the current step
            loss = criterion(predictions, landmarks)

            loss_valid += loss.item()

            pred_numpy = predictions.cpu().detach().numpy()
            pred_numpy = pred_numpy.reshape(pred_numpy.shape[0], -1, 2)

            land_numpy = landmarks.cpu().detach().numpy()
            land_numpy = land_numpy.reshape(pred_numpy.shape[0], -1, 2)

            all_land = np.vstack([all_land, rescale_landmark(land_numpy)])
            all_pred = np.vstack([all_pred, rescale_landmark(pred_numpy)])
        
    assert(all_land.shape == (len(valid_loader.dataset), 68, 2))
    assert(all_pred.shape == (len(valid_loader.dataset), 68, 2))
    score_valid += score(all_land, all_pred)
    loss_valid /= len(valid_loader)
    print(f'Eval set: Average loss: {loss_valid} Average score: {score_valid}')
    return loss_valid, score_valid


def visualize(valid_loader, model, save_path):
    with torch.no_grad():
        model.eval()
        
        images, landmarks = next(iter(valid_loader))
        
        images = images.to(DEVICE)
        landmarks = rescale_landmark(landmarks)

        predictions =rescale_landmark(model(images).cpu())
        predictions = predictions.view(-1,68,2)
        
        plt.figure(figsize=(10,40))
        
        for img_num in range(8):
            plt.subplot(8,1,img_num+1)
            plt.imshow(images[img_num].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num,:,0], predictions[img_num,:,1], c = 'r', s = 5)
            plt.scatter(landmarks[img_num,:,0], landmarks[img_num,:,1], c = 'g', s = 5)

        plt.savefig(save_path)
        plt.show()


def check_modelsize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def generate_PDB_annot(args):

    print('start to generate  Pose-based Data Balanced annot.pkl')
    image_filenames, landmarks = None, None
    with open(os.path.join(args.dataroot, 'annot.pkl'), 'rb') as f:
        annot = pickle.load(f)
        image_filenames, landmarks = annot

    dataset_lenth=len(image_filenames)
    landmarks_origin=landmarks

    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(-1, 136)
    pca = PCA(n_components=1)
    pca.fit(landmarks)
    trans_landmarks = pca.transform(landmarks).flatten()

    mean = np.mean(trans_landmarks)
    std = np.std(trans_landmarks)

    ## save histogram for PCA
    # plt.hist(trans_landmarks)
    # plt.savefig('./distribution.png')
    # plt.show()
    # print(f'origin datasize: {dataset_lenth}')

    # Pose-based Data Balanced
    for idx in tqdm(range(dataset_lenth)):
        if mean - std < trans_landmarks[idx] < mean + std:
            image_filenames.append(image_filenames[idx])
            landmarks_origin.append(landmarks_origin[idx])
        # if trans_landmarks[idx] > mean + 2*std:
        #     image_filenames.append(image_filenames[idx])
        #     image_filenames.append(image_filenames[idx])
        #     landmarks_origin.append(landmarks_origin[idx])
        #     landmarks_origin.append(landmarks_origin[idx])

        # elif trans_landmarks[idx] > mean + std:
        #     image_filenames.append(image_filenames[idx])
        #     landmarks_origin.append(landmarks_origin[idx])

        # elif trans_landmarks[idx] < mean - 2*std:
        #     image_filenames.append(image_filenames[idx])
        #     image_filenames.append(image_filenames[idx])
        #     landmarks_origin.append(landmarks_origin[idx])
        #     landmarks_origin.append(landmarks_origin[idx])

        # elif trans_landmarks[idx] < mean - std:
        #     image_filenames.append(image_filenames[idx])
        #     landmarks_origin.append(landmarks_origin[idx])
        
        # else: continue

    # print(f'augment datasize: {len(image_filenames)}')

    #  generate Pose-based Data Balanced annot.pkl
    
    with open(os.path.join(args.dataroot, 'PDB_annot.pkl'), 'wb') as f: 
        pickle.dump((image_filenames,landmarks_origin), f)


    print('origin image len: ', dataset_lenth)
    print('PDB image len: ', len(image_filenames))



def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = False  
    torch.backends.cudnn.benchmark = False  
    
def validate_each(valid_loader, model, criterion):
    model.eval()
    score_valid, loss_valid = 0, 0

    all_land = np.array([]).reshape(0, 68, 2)
    all_pred = np.array([]).reshape(0, 68, 2)
    with torch.no_grad():
        for img, landmarks in valid_loader:
            img = img.to(DEVICE)
            landmarks = landmarks.view(landmarks.size(0),-1).cuda()
            predictions = model(img)

            # find the loss for the current step
            loss = criterion(predictions, landmarks)

            loss_valid += loss.item()

            pred_numpy = predictions.cpu().detach().numpy()
            pred_numpy = pred_numpy.reshape(pred_numpy.shape[0], -1, 2)

            land_numpy = landmarks.cpu().detach().numpy()
            land_numpy = land_numpy.reshape(pred_numpy.shape[0], -1, 2)

            all_land = np.vstack([all_land, rescale_landmark(land_numpy)])
            all_pred = np.vstack([all_pred, rescale_landmark(pred_numpy)])
        
    assert(all_land.shape == (len(valid_loader.dataset), 68, 2))
    assert(all_pred.shape == (len(valid_loader.dataset), 68, 2))
    score_valid += score(all_land, all_pred)
    loss_valid /= len(valid_loader)
    print(f'Eval set: Average loss: {loss_valid} Average score: {score_valid}')

    score_valid_each = []
    for i in range(all_land.shape[0]):
        score_valid_each.append(score_each(all_land[i], all_pred[i]))

    return np.array(score_valid_each)
