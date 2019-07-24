import glob
import pandas as pd
import os
import sys
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

from data_process.data_utils import *

# sys.path.append('../')


def data_filter(root, data_ids, p_keep):
    filtered_ids = []
    for mask_id in data_ids:
        mask = cv2.imread(os.path.join(root, f'{mask_id}.png'), 0)
        if np.sum(mask > 0) > 0 or np.random.rand() < p_keep:
            filtered_ids.append(mask_id)
    return np.array(filtered_ids)


class SIIMDataset(Dataset):
    def __init__(self, root='data_process/input/data', transform=None, subset='train', image_size=512,
                folds_dir='data_process/splits/10folds', fold_id=0, prob_keep=None, coord_conv=True):
        """

        :param root: string -- the data directory
        :param transform: function -- the transformers of the data
        :param subset: string -- the name of subset, which can be `train`, `test`, `valid`
        :param image_size: int -- the size of images
        :param folds_dir: sting -- the folder's path, which contains the splitted data
        :param fold_id:
        :param prob_keep:
        """
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
        self.coord_conv = coord_conv

        self.coord_vec_row = np.expand_dims(np.array([[ii] * image_size for ii in range(image_size)]), axis=0)
        self.coord_vec_column = np.expand_dims(np.array([list(range(image_size)) for _ in range(image_size)]), axis=0)

        suff = '_{}'.format(image_size)
        if self.subset in ['train', 'valid']:
            self.img_dir = os.path.join(self.root, 'train' + suff)
            self.label_dir = os.path.join(self.root, 'train_mask' + suff)
            csv_path = os.path.join(folds_dir, f'fold{fold_id}_{self.subset}.csv')
            self.img_list = np.array(pd.read_csv(csv_path)['Folds'])
            if prob_keep is not None:
                self.img_list = data_filter(root=self.label_dir, data_ids=self.img_list, p_keep=prob_keep)

            features_df = pd.read_csv(os.path.join(self.root, 'train_features.csv'))
            features_df = features_df.to_dict('records')
            for row in features_df:
                self.features_dict[row['ImageId']] = row
        
        else:
            self.img_dir = os.path.join(self.root, 'test' + suff)
            features_df = pd.read_csv(os.path.join(self.root, 'test_features.csv'))
            self.img_list = np.array(features_df['ImageId'])
            for _, row in features_df.iterrows():
                try:
                    self.features_dict[row['ImeageId']] = row.to_dict()
                except Exception as e:
                    pass

    def __getitem__(self, index):
        
        # load image and labels
        img_id = self.img_list[index]
        img = cv2.imread(self.img_dir + f'/{img_id}.png', 0)
        target = cv2.imread(self.label_dir + f'/{img_id}.png', 0) if not self.subset == 'test' else None

        # apply transforms to both
        if target is not None:
            img, target = self.transform({'input': img, 'mask': target}).values()
        else:
            img, _ = self.transform({'input': img, 'mask': np.zeros(img.shape)}).values()

        if self.coord_conv:
            img = np.concatenate((np.expand_dims(img, axis=0), self.coord_vec_row, self.coord_vec_column), axis=0)

        if target is not None:
            # assert target.max() == 1.0, 'Wrong scaling for target mask (max val = {})'.format(target.max())
            target[(target > 0) & (target < 1.0)] = 0
            # assert ((target > 0) & (target < 1.0)).sum() == 0
            return {'input': torch.Tensor(img), 'target': torch.Tensor(target), 'params': self.features_dict[img_id]}
        else:
            return {'input': torch.Tensor(img), 'params': self.features_dict[img_id]}

    def __len__(self):
        return len(self.img_list)
