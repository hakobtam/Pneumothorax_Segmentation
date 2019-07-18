import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')
import glob
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample


def random_split(data_dir, n_folds, seed=42):
    print('Generating random {} fold splits'.format(n_folds))
    out_path = 'data_process/splits/{}folds'.format(n_folds)
    os.makedirs(out_path, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    filenames = np.array([os.path.basename(f)[:-4] for f in files])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (tr_ids, val_ids) in enumerate(kf.split(np.arange(len(filenames)))):
        tr_fold_dict = {'Folds': filenames[tr_ids]}
        val_fold_dict = {'Folds': filenames[val_ids]}
        pd.DataFrame(tr_fold_dict).to_csv('{}/fold{}_train.csv'.format(out_path, i))
        pd.DataFrame(val_fold_dict).to_csv('{}/fold{}_valid.csv'.format(out_path, i))

def stratified_split(data_dir, n_folds, rs):
    #TODO
    pass

def replacement_sampling(data_dir, n_folds, rs):
    #TODO
    pass

if __name__ == '__main__':
        random_split('../input/data/train_256', n_folds=10, seed=32)
