import numpy as np
from tqdm import tqdm
from data_process.data_utils import *
import pandas as pd

threshold = 0.501
prob_filename = 'Aug15_22-29-05_Unet_best-dice_22_tta'
prefix = prob_filename.endswith('tta')
sub_filename = prob_filename.split('_tta')[0] + f'_{threshold}' + '_tta' if prefix else prob_filename.split + f'_{threshold}'
outputs = np.load(f'predictions/{prob_filename}.npy', allow_pickle=True)
#print(dict(outputs)['ImageId'])
outputs = outputs[()]
noise_th=100.0*(1024/128.0)**2

submission = {'ImageId': [], 'EncodedPixels': []}
for image_id, probs in tqdm(zip(outputs['ImageId'], outputs['Predictions'])):
    preds = probs > (threshold * 65535)
    preds[preds.sum((0,1)) < noise_th] = 0.0
    rle = mask2rle((preds.T * 255).astype(np.uint8), 1024, 1024)
    submission['ImageId'].append(image_id)
    submission['EncodedPixels'].append(rle)

sub_df = pd.DataFrame(submission)
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
print(f'Submission saved in submissions/{sub_filename}.csv file!')
sub_df.to_csv(f'submissions/{sub_filename}.csv', index=False)

