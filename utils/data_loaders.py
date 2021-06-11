from __future__ import print_function
import csv
import glob
import os
import cv2
from shutil import copy2

from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import parse_args

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, idx):

        N_WAY = self.args.N_way
        K_SHOT = self.args.K_shot
        QUERY_NUM = self.args.query_num
        IMG_SIZE = self.img_size
        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        # Sample classes
        sample_cls = random.sample(self.classes, N_WAY)

        # Sample images from classes
        sample_img = {}
        for cls in sample_cls:
            img_dir = self.images[cls]
            imgs = random.sample(img_dir, K_SHOT + QUERY_NUM)
            sample_img[cls] = imgs

        # Support / Query
        support_x = np.zeros([N_WAY, K_SHOT, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        support_y = np.zeros([N_WAY, K_SHOT]).astype(int)
        query_x = np.zeros([N_WAY, QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        query_y = np.zeros([N_WAY, QUERY_NUM]).astype(int)

        # Support / Query
        for cls_idx, cls in enumerate(sample_cls, 0):
            img_names = sample_img[cls]
            for img_idx, img_name in enumerate(img_names, 0):
                img_dir = os.path.join(DATA_DIR, cls, img_name)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img_idx < K_SHOT:
                    support_x[cls_idx, img_idx] = img
                    support_y[cls_idx, img_idx] = cls_idx
                else:
                    query_x[cls_idx, img_idx-K_SHOT] = img
                    query_y[cls_idx, img_idx-K_SHOT] = cls_idx

        # Shuffle Support / Query
        support_x = np.reshape(support_x, [N_WAY*K_SHOT, IMG_SIZE, IMG_SIZE, 3])
        support_y = np.reshape(support_y, [N_WAY*K_SHOT])
        query_x = np.reshape(query_x, [N_WAY*QUERY_NUM, IMG_SIZE, IMG_SIZE, 3])
        query_y = np.reshape(query_y, [N_WAY*QUERY_NUM])

        sup_idx = np.arange(N_WAY*K_SHOT)
        query_idx = np.arange(N_WAY*QUERY_NUM)  
        np.random.shuffle(sup_idx)
        np.random.shuffle(query_idx)

        # Support x : [N_WAY * K_SHOT, IMG_SIZE, IMG_SIZE, 3]
        # Support y : [N_WAY * K_SHOT]
        # Query x : [N_WAY * QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]
        # Query y : [N_WAY * QUERY_NUM]
        support_x = support_x[sup_idx]
        support_y = support_y[sup_idx]
        query_x = query_x[query_idx]
        query_y = query_y[query_idx]

        support_x = ((torch.from_numpy(support_x)).permute(0,3,1,2)).float()
        support_y = torch.from_numpy(support_y)
        query_x = (torch.from_numpy(query_x).permute(0,3,1,2)).float()
        query_y = torch.from_numpy(query_y)

        return support_x, support_y, query_x, query_y

    def __len__(self):
        return 10000

def proc_images(args):
    DATA_DIR = args.data_dir
    DATASET = args.dataset

    path_to_images = os.path.join(DATA_DIR, DATASET, 'images/')

    all_images = glob.glob(path_to_images + '*')

    # Resize images
    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 500 == 0:
            print(i)

    # Put in correct directory
    for datatype in ['train', 'val', 'test']:

        dir = os.path.join(DATA_DIR, DATASET, datatype)

        os.system('mkdir ' + dir)

        with open(dir + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            last_label = ''
            for i, row in enumerate(reader):
                if i == 0:  # skip the headers
                    continue
                label = row[1]
                image_name = row[0]
                if label != last_label:
                    cur_dir = dir + '/' + label + '/'
                    if not os.path.exists(cur_dir):
                        os.mkdir(cur_dir)
                    last_label = label
                copy2(path_to_images + image_name, cur_dir)


def show_samples(args, subset):

    TASK_NUM = args.task_num
    N_WAY = args.N_way
    K_SHOT = args.K_shot
    QUERY_NUM = args.query_num
    IMG_SIZE = 84
    PLAYGROUND_DIR = 'playground/'

    dataset = Dataset(args, subset)
    dataloader = DataLoader(dataset, batch_size=TASK_NUM, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        support_x, support_y, query_x, query_y = data

        support_x = (support_x.permute(0,1,3,4,2)).numpy()
        support_y = support_y.numpy()
        query_x = (query_x.permute(0,1,3,4,2)).numpy()
        query_y = query_y.numpy()

        BATCH_SIZE = len(support_x)

        for b_idx in range(BATCH_SIZE):
            cur_sup_x = support_x[b_idx]
            cur_sup_y = support_y[b_idx]
            cur_que_x = query_x[b_idx]
            cur_que_y = query_y[b_idx]

            show_sup = np.zeros([N_WAY*IMG_SIZE, K_SHOT*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(K_SHOT):
                    idx = K_SHOT * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_sup_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_sup_y[idx]
                    cv2.putText(cur_img, str(cur_label), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    show_sup[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            show_query = np.zeros([N_WAY*IMG_SIZE, QUERY_NUM*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(QUERY_NUM):
                    idx = QUERY_NUM * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_que_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_que_y[idx]
                    cv2.putText(cur_img, str(cur_label), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    show_query[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            img_name = PLAYGROUND_DIR +  str(i).zfill(3) + '_' + str(b_idx) + '_'

            cv2.imwrite(img_name + 'support.jpg', show_sup)
            cv2.imwrite(img_name + 'query.jpg', show_query)   

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    subset = 'train'

    show_samples(args, subset)