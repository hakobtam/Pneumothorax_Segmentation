import pydicom
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')
import PIL
import cv2
import glob
from tqdm import tqdm

from data_utils import rle2mask, mask2rle

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
    inp_path = '../input/dicom-images-{}'.format(dataset)
    out_path = '../input/data'
    files = glob.glob(os.path.join(inp_path, '**', '*.dcm'), recursive=True)
    for file_path in tqdm(files):
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
    return pd.DataFrame(data_dict).to_csv('{}/{}_features.csv'.format(out_path, dataset))

def dcm2png(SZ, dataset):
    inp_path = '../input/dicom-images-{}'.format(dataset)
    out_path = '../input/data/{}_{}'.format(dataset, SZ)
    os.makedirs(out_path, exist_ok=True)
    files = glob.glob(os.path.join(inp_path, '**', '*.dcm'), recursive=True)
    for f in tqdm(files):
        dcm = pydicom.read_file(str(f)).pixel_array
        img = cv2.resize(dcm, (SZ, SZ), interpolation=Interpolation) if SZ != 1024 else dcm
        cv2.imwrite('{}/{}.png'.format(out_path, os.path.basename(f)[:-4]), img)

def masks2png(SZ):
    out_path = '../input/data/train_mask_{}'.format(SZ)
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
            
            img = cv2.resize(m, (SZ, SZ), interpolation=Interpolation) if SZ != 1024 else m
            cv2.imwrite('{}/{}.png'.format(out_path, name), img)
        else:
            m = np.array([rle2mask(e, 1024, 1024).astype(np.uint8) for e in enc.values])
            m = m.sum(0).astype(np.uint8).T
            img = cv2.resize(m, (SZ, SZ), interpolation=Interpolation) if SZ != 1024 else m
            cv2.imwrite('{}/{}.png'.format(out_path, name.values[0]), img)

rle_df = pd.read_csv('../input/train-rle.csv')
Interpolation = cv2.INTER_LINEAR # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4

for SZ in [64, 128, 256, 512, 1024]:
    print('Converting data for train{}'.format(SZ))
    dcm2png(SZ, 'train')
    print('Converting data for test{}'.format(SZ))
    dcm2png(SZ, 'test')
    print('Generating masks for size {}'.format(SZ))
    masks2png(SZ)

for SZ in [64, 128, 256, 512, 1024]:
    # Missing masks set to 0
    print('Generating missing masks as zeros')
    train_images = [os.path.basename(f) for f in glob.glob('../input/data/train_{}/*.png'.format(SZ))]
    train_masks = [os.path.basename(f) for f in glob.glob('../input/data/train_mask_{}/*.png'.format(SZ))]
    missing_masks = set(train_images) - set(train_masks)
    for name in tqdm(missing_masks):
        m = np.zeros((1024, 1024)).astype(np.uint8).T
        img = cv2.resize(m, (SZ, SZ), interpolation=Interpolation) if SZ != 1024 else m
        cv2.imwrite('../input/data/train_mask_{}/{}'.format(SZ, name), img)

print('Converting data remaining features for train')
dcm2csv('train')
print('Converting data remaining features for test')
dcm2csv('test')
