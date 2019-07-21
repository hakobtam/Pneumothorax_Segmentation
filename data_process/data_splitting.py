import numpy as np
import pandas as pd
import os
import sys
import glob
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample


sys.path.append('../')


def random_split(data_dir, n_folds, seed=42):
    print(f'Generating random {n_folds} fold splits')
    out_path = f'data_process/splits/{n_folds}folds'
    os.makedirs(out_path, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    filenames = np.array([os.path.basename(f)[:-4] for f in files])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (tr_ids, val_ids) in enumerate(kf.split(np.arange(len(filenames)))):
        tr_fold_dict = {'Folds': filenames[tr_ids]}
        val_fold_dict = {'Folds': filenames[val_ids]}
        pd.DataFrame(tr_fold_dict).to_csv(f'{out_path}/fold{i}_train.csv')
        pd.DataFrame(val_fold_dict).to_csv(f'{out_path}/fold{i}_valid.csv')


def stratified_split(data_dir, n_folds, rs):
    # TODO
    pass


def replacement_sampling(data_dir, n_folds, rs):
    # TODO
    pass


if __name__ == '__main__':
        random_split('../input/data/train_256', n_folds=10, seed=32)
