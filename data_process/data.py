import random
import glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import sys
sys.path.append('../')

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data_process.data_utils import *

def data_filter(root, data_ids, p_keep):
    filtered_ids = []
    for mask_id in data_ids:
        mask = cv2.imread(root + f'/{mask_id}.png', 0)
        if np.sum(mask > 0) > 0 or np.random.rand() < p_keep:
            filtered_ids.append(mask_id)
    return np.array(filtered_ids)

class SIIMDataset(Dataset):

    def __init__(self, root='../input/data', transform=None, subset='train', image_size=512, 
                folds_dir='./splits/10folds', fold_id=0, prob_keep=None):
        
        assert subset in ['train', 'valid', 'test'], 'Unknown subset: {}'.format(subset)
        num_folds = len(glob.glob(folds_dir + '/*'))
        assert num_folds % 2 == 0
        assert 0 <= fold_id < num_folds/2, fold_id
        print('SIIMDataset::folds version={}'.format(os.path.basename(folds_dir)))

        self.root = root
        self.transform = transform
        self.subset = subset
        self.features_dict = {}
        self.img_dir, self.label_dir = None, None
        self.img_list = []
        
        if self.subset in ['train', 'valid']:
            suff = '_{}'.format(image_size)
            self.img_dir = self.root + '/train' + suff
            self.label_dir = self.root + '/train_mask' + suff
            csv_path = folds_dir + f'/fold{fold_id}_{self.subset}.csv'
            self.img_list = np.array(pd.read_csv(csv_path)['Folds'])
            if prob_keep is not None:
                self.img_list = data_filter(root=self.label_dir, data_ids=self.img_list, p_keep=prob_keep)

            features_df = pd.read_csv(self.root + '/train_features.csv')
            for _, row in features_df.iterrows():
                self.features_dict[row['ImeageId']] = row.to_dict()
        
        else:
            self.img_dir = self.root + '/test' + suff
            features_df = pd.read_csv(self.root + '/test_features.csv')
            self.img_list = np.array(features_df['ImageId'])
            for _, row in features_df.iterrows():
                self.features_dict[row['ImeageId']] = row.to_dict()

    def __getitem__(self, index):
        
        # load image and labels
        img_id = self.img_list[index]
        img = cv2.imread(self.img_dir + f'/{img_id}.png', 0)
        target =cv2.imread(self.label_dir + f'/{img_id}.png', 0) if not self.subset == 'test' else None

        # apply transforms to both
        if target is not None:
            img, target = self.transform({'input': img, 'mask':target}).values()
        else:
            img, _ = self.transform({'input': img, 'mask': np.zeros(img.shape)}).values()

        if target is not None:
            assert target.max() == 1.0, 'Wrong scaling for target mask (max val = {})'.format(target.max())
            target[(target > 0) & (target < 1.0)] = 0
            assert ((target > 0) & (target < 1.0)).sum() == 0
            return {'input': torch.Tensor(img), 'target': torch.Tensor(target), 'params': self.features_dict[img_id]}
        else:
            return {'input': torch.Tensor(img), 'params': self.features_dict[img_id]}

    def __len__(self):
        return len(self.img_list)
