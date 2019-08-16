import time
import argparse
from tqdm import tqdm

import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *

import models
from data_process import SIIMDataset
from data_process.data_utils import *
from utils import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--use_gpu", action='store_true', help='freeze encoder weights')
parser.add_argument("--num_workers", type=int, default=4, help='number of workers')
parser.add_argument('--seed', type=int, default=27, help='seed value for random state')
parser.add_argument("--threshold", type=float, default=0.5, help='probability threshold')
parser.add_argument('--tta', action='store_true', help='test time augmentation')
parser.add_argument('--ckpt', type=str, help='location model checkpoint')
args = parser.parse_args()

orig_img_size = 1024
img_size = 512

use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print('use_gpu', use_gpu)

print("loading model...")
checkpoint = torch.load(args.ckpt)
filenames = args.ckpt.split('/')
sub_filename = os.path.join('submissions', filenames[-3] + '_' + filenames[-1].split('.pt')[0] + f'_{args.threshold}')
prob_filename = os.path.join('predictions', filenames[-3] + '_' + filenames[-1].split('.pt')[0])
if args.tta:
   sub_filename += '_tta'
   prob_filename += '_tta'

preprocessing_fn = models.encoders.get_preprocessing_fn(checkpoint['encoder'], checkpoint['pretrained'])
model = getattr(models, checkpoint['model'])(
                                            encoder_name=checkpoint['encoder'],
                                            encoder_weights=checkpoint['pretrained'], 
                                            classes=1, 
                                            activation='sigmoid'
                                        )
model.load_state_dict(checkpoint['model_state'])

if use_gpu:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    model.cuda()

def predict(model, batch, flipped_batch, use_gpu):
    inputs = batch['input']
    if use_gpu:
        inputs = inputs.cuda()
    outputs = model(inputs)
    probs = torch.sigmoid(outputs)

    if flipped_batch is not None:
        flipped_inputs = flipped_batch['input']
        if use_gpu:
            flipped_inputs = flipped_inputs.cuda()
        flipped_outputs = model(flipped_inputs)
        flipped_probs = torch.sigmoid(flipped_outputs)
        probs += torch.flip(flipped_probs, (3,))  # flip back and add
        probs *= 0.5

    probs = probs.squeeze(1).cpu().numpy()
    probs = np.swapaxes(probs, 0, 2)
    probs = cv2.resize(probs, (orig_img_size, orig_img_size), interpolation=cv2.INTER_LINEAR)
    probs = np.swapaxes(probs, 0, 2)
    return probs

def test():
    test_transform = Compose([PrepareData()])
    preprocessing_transform = Compose([preprocessing_fn])

    test_dataset =SIIMDataset(subset='test', transform=test_transform, 
                    preprocessing=preprocessing_transform, img_size=checkpoint['img_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    flipped_test_transform = Compose([PrepareData(), 
                                      HorizontalFlip()])
    flipped_test_dataset = SIIMDataset(subset='test', transform=flipped_test_transform, 
                                preprocessing=preprocessing_transform, img_size=checkpoint['img_size'])
    flipped_test_dataloader_iter = iter(DataLoader(flipped_test_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers))

    noise_th=75.0*(checkpoint['img_size']/128.0)**2
    model.eval()
    torch.set_grad_enabled(False)

    prediction = {'ImageId': [], 'Predictions': []}
    submission = {'ImageId': [], 'EncodedPixels': []}
    pbar = tqdm(test_dataloader, unit="images", unit_scale=test_dataloader.batch_size, disable=None)

    empty_images_count = 0
    for batch in pbar:
        if args.tta:
            flipped_batch = next(flipped_test_dataloader_iter)
        else:
            flipped_batch = None

        probs = predict(model, batch, flipped_batch, use_gpu=use_gpu)
        preds = probs > args.threshold
        preds[preds.sum((1,2)) < noise_th] = 0.0
        empty_images_count += (preds.sum(axis=(1, 2)) == 0).sum()

        probs_uint16 = (65535 * probs).astype(dtype=np.uint16)
        image_ids = batch['params']['ImageId']
        prediction['ImageId'] += image_ids
        prediction['Predictions'] += list(probs_uint16)
        #prediction.update(dict(zip(image_ids, probs_uint16)))
        rles = [mask2rle((pred.T * 255).astype(np.uint8), 1024, 1024) for pred in preds]
        submission['ImageId'] += image_ids
        submission['EncodedPixels'] += rles
        #submission.update(dict(zip(image_ids, rle)))

    # empty_images_percentage = empty_images_count / len(image_ids)
    # print("empty images: %.2f%% (in public LB 38%%)" % (100 * empty_images_percentage))

    #gzip_save('-'.join([args.output_prefix, 'probabilities.pkl.gz']), prediction)
    print(f'Predictions saved in {prob_filename}.npy file!')
    np.save(f'{prob_filename}.npy', prediction)
    sub_df = pd.DataFrame(submission)
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    print(f'Submission saved in {sub_filename}.csv file!')
    sub_df.to_csv(f'{sub_filename}.csv', index=False)


if __name__ == '__main__':

    print("testing {}...".format(args.ckpt))
    since = time.time()
    test()
    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                        time_elapsed % 60)
    print("finished")
