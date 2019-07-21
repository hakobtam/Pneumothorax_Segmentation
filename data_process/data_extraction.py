import pydicom
import pandas as pd
import numpy as np
import os
import sys
import PIL
import glob
from tqdm import tqdm

from data_process.data_utils import rle2mask, mask2rle


dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(dir_path, "input")


def dcm2csv(dataset):
    data_dict = {
                'ImageId': [],
                'PatientId': [],
                'Age': [],
                'Gender': [],
                'Position': [],
                'ImgSize': [],
                'PixelSpacing': []
              }
    inp_path = os.path.join(dir_path, 'dicom-images-{0}'.format(dataset))
    out_path = os.path.join(dir_path, 'data')
    files = glob.glob(os.path.join(inp_path, '**', '*.dcm'), recursive=True)
    for file_path in tqdm(files):
        if os.path.isfile(file_path):
            data = pydicom.dcmread(file_path)
            data_dict['ImageId'].append(os.path.basename(file_path)[:-4])
            data_dict['PatientId'].append(data.PatientID)
            data_dict['Age'].append(data.PatientAge)
            data_dict['Gender'].append(data.PatientSex)
            data_dict['Position'].append(data.ViewPosition)
            if 'PixelData' in data:
                data_dict['ImgSize'].append((int(data.Rows), int(data.Columns)))
                data_dict['PixelSpacing'].append(data.PixelSpacing)
            else:
                data_dict['ImgSize'].append(None)
                data_dict['PixelSpacing'].append(None)
    return pd.DataFrame(data_dict).to_csv(os.path.join(out_path, f'{dataset}_features.csv'))


def dcm2png(SZ, dataset):
    inp_path = os.path.join(dir_path, 'dicom-images-{0}'.format(dataset))
    out_path = os.path.join(dir_path, 'data', '{0}_{1}'.format(dataset, SZ))
    os.makedirs(out_path, exist_ok=True)
    files = glob.glob(os.path.join(inp_path, '**', '*.dcm'), recursive=True)
    for f in tqdm(files):
        if os.path.isfile(f):
            dcm = pydicom.dcmread(f).pixel_array
            PIL.Image.fromarray(dcm).resize((SZ, SZ)).save(os.path.join(out_path, f'{os.path.basename(f)[:-4]}.png'))


def masks2png(SZ):
    out_path = os.path.join(dir_path, 'data', 'train_mask_{0}'.format(SZ))
    os.makedirs(out_path, exist_ok=True)
    for i in tqdm(list(set(rle_df.ImageId.values))):
        I = rle_df.ImageId == i
        name = rle_df.loc[I, 'ImageId']
        enc = rle_df.loc[I, ' EncodedPixels']
        if sum(I) == 1:
            enc = enc.values[0]
            name = name.values[0]
            if enc == ' -1':
                m = np.zeros((1024, 1024)).astype(np.uint8).T
            else:
                m = rle2mask(enc, 1024, 1024).astype(np.uint8).T
            PIL.Image.fromarray(m).resize((SZ, SZ)).save(os.path.join(out_path, f'{name}.png'))
        else:
            m = np.array([rle2mask(e, 1024, 1024).astype(np.uint8) for e in enc.values])
            m = m.sum(0).astype(np.uint8).T
            PIL.Image.fromarray(m).resize((SZ, SZ)).save(os.path.join(out_path, f'{name.values[0]}.png'))


rle_df = pd.read_csv(os.path.join(dir_path, 'train-rle.csv'))


for SZ in [64, 128, 256, 512, 1024]:
    print(f'Converting data for train{SZ}')
    dcm2png(SZ, 'train')
    print(f'Converting data for test{SZ}')
    dcm2png(SZ, 'test')
    print(f'Generating masks for size {SZ}')
    masks2png(SZ)


for SZ in [64, 128, 256, 512, 1024]:
    # Missing masks set to 0
    print('Generating missing masks as zeros')
    train_images = [os.path.basename(f) for f in glob.glob(os.path.join(dir_path, 'data', 'train_{0}\*.png'.format(SZ)))]
    train_masks = [os.path.basename(f) for f in glob.glob(os.path.join(dir_path, 'data', 'train_mask_{0}\*.png'.format(SZ)))]
    missing_masks = set(train_images) - set(train_masks)
    for name in tqdm(missing_masks):
        m = np.zeros((1024, 1024)).astype(np.uint8).T
        PIL.Image.fromarray(m).resize((SZ, SZ)).save(os.path.join(dir_path, 'data', 'train_mask_{0}\{1}'.format(SZ, name)))


print('Converting data remaining features for train')
dcm2csv('train')
print('Converting data remaining features for test')
dcm2csv('test')
