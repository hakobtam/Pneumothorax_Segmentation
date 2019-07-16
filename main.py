import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn


from data_processor import PneumothoraxDataset
from train import train_model, eval_model
from models import UNet

dcm_train_dir = r"C:\Users\Grisha Zohrabyan\dev\Kaggle\pneumothorax_data\images-train"
dcm_test_dir = r"C:\Users\Grisha Zohrabyan\dev\Kaggle\pneumothorax_data\images-test"
masks_csv = r"C:\Users\Grisha Zohrabyan\dev\Kaggle\pneumothorax_data\train-rle.csv"

train_set = PneumothoraxDataset(dcm_images_dir=dcm_train_dir, mask_csv_path=masks_csv)
test_set = PneumothoraxDataset(dcm_images_dir=dcm_test_dir, mask_csv_path=masks_csv)

epochs = 10
batch_size = 32
learning_rate = 0.0001
shuffle = True
num_workers = 1

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
model = UNet(1, 1)
optimizer = Adam(params=model.parameters(), lr=learning_rate)


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y.double() - y_hat.double()).pow(2)))


def rle_loss(y, y_hat):

    def array2rle(arr):
        pass



loss = rmse

if __name__ == '__main__':
    for e in range(epochs):
        train_model(model=model, optimizer=optimizer, data=train_dataloader, loss_fn=loss, epoch=e)

