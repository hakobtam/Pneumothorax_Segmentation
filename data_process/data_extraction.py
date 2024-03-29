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

from data_process.data_utils import rle2mask, mask2rle

def remove_files():
    train_files = glob.glob('../input/dicom-images-train/**/*.dcm', recursive=True)
    test_files = glob.glob('../input/dicom-images-test/**/*.dcm', recursive=True)
    train_removes = [
        '1.2.276.0.7230010.3.1.4.8323329.11566.1517875233.640521',
        '1.2.276.0.7230010.3.1.4.8323329.11104.1517875231.169401',
        '1.2.276.0.7230010.3.1.4.8323329.31801.1517875156.929061',
        '1.2.276.0.7230010.3.1.4.8323329.11584.1517875233.731531',
        '1.2.276.0.7230010.3.1.4.8323329.11557.1517875233.601090',
        '1.2.276.0.7230010.3.1.4.8323329.3352.1517875177.433385',
        '1.2.276.0.7230010.3.1.4.8323329.14557.1517875252.690062',
        '1.2.276.0.7230010.3.1.4.8323329.4373.1517875182.554858',
        '1.2.276.0.7230010.3.1.4.8323329.2563.1517875173.431928',
        '1.2.276.0.7230010.3.1.4.8323329.12062.1517875237.179186',
        '1.2.276.0.7230010.3.1.4.8323329.4468.1517875183.20323',
        '1.2.276.0.7230010.3.1.4.8323329.4843.1517875185.73985',
        '1.2.276.0.7230010.3.1.4.8323329.10231.1517875222.737143',
        '1.2.276.0.7230010.3.1.4.8323329.10407.1517875223.567351',
        '1.2.276.0.7230010.3.1.4.8323329.3089.1517875176.36192',
        '1.2.276.0.7230010.3.1.4.8323329.11577.1517875233.694347',
        '1.2.276.0.7230010.3.1.4.8323329.2309.1517875172.75133',
        '1.2.276.0.7230010.3.1.4.8323329.4134.1517875181.277174',
        '1.2.276.0.7230010.3.1.4.8323329.13415.1517875245.218707',
        '1.2.276.0.7230010.3.1.4.8323329.10599.1517875224.488727',
        '1.2.276.0.7230010.3.1.4.8323329.1068.1517875166.144255',
        '1.2.276.0.7230010.3.1.4.8323329.13620.1517875246.884737',
        '1.2.276.0.7230010.3.1.4.8323329.4996.1517875185.888529',
        '1.2.276.0.7230010.3.1.4.8323329.5278.1517875187.330082',
        '1.2.276.0.7230010.3.1.4.8323329.2630.1517875173.773726',
        '1.2.276.0.7230010.3.1.4.8323329.3714.1517875179.128897',
        '1.2.276.0.7230010.3.1.4.8323329.5543.1517875188.726955',
        '1.2.276.0.7230010.3.1.4.8323329.3321.1517875177.247887',
        '1.2.276.0.7230010.3.1.4.8323329.10362.1517875223.377845',
        '1.2.276.0.7230010.3.1.4.8323329.2187.1517875171.557615',
        '1.2.276.0.7230010.3.1.4.8323329.3791.1517875179.436805',
        '1.2.276.0.7230010.3.1.4.8323329.5087.1517875186.354925',
        '1.2.276.0.7230010.3.1.4.8323329.32688.1517875161.809571',
        '1.2.276.0.7230010.3.1.4.8323329.11215.1517875231.757436',
        '1.2.276.0.7230010.3.1.4.8323329.32302.1517875159.778024',
        '1.2.276.0.7230010.3.1.4.8323329.2083.1517875171.71387',
        '1.2.276.0.7230010.3.1.4.8323329.13378.1517875244.961609'      
    ]
    test_removes = [
        '1.2.276.0.7230010.3.1.4.8323329.6491.1517875198.577052', 
        '1.2.276.0.7230010.3.1.4.8323329.7013.1517875202.343274', 
        '1.2.276.0.7230010.3.1.4.8323329.6370.1517875197.841736', 
        '1.2.276.0.7230010.3.1.2.8323329.7020.1517875202.386064',
        '1.2.276.0.7230010.3.1.4.8323329.6082.1517875196.407031'
    ]
    print("Removing {} additional train and {} test files".format(len(train_removes), len(test_removes)))
    for f in train_files:
        if os.path.basename(f)[:-4] in train_removes:
            os.rename(f, f[:-4] + '.remove')
    for f in test_files:
        if os.path.basename(f)[:-4] in test_removes:
            os.rename(f, f[:-4] + '.remove')

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

if __name__ == '__main__':
    rle_df = pd.read_csv('../input/train-rle.csv')
    Interpolation = cv2.INTER_LINEAR # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4

    #Removing additional imgs
    remove_files()

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
