import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import random

from models import UNet
from data_process import SIIMDataset
from torchvision import transforms

from data_process import PrepareData, HWCtoCHW, mask2rle


# TODO: add transformers, -- done, but
# TODO: add LovaSz-softmax loss  -- done, but need bug fixing
# TODO: change accuracy's calculation -- not done

epochs = 10
batch_size = 32
learning_rate = 0.00001
shuffle = True
num_workers = 1

model = UNet(3, 1)
optimizer = Adam(params=model.parameters(), lr=learning_rate)


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y.double() - y_hat.double()).pow(2)))


def calculate_accuracy(x, y):
    x = x.squeeze(dim=1)
    y = y.squeeze(dim=1)

    zero_matrix = torch.zeros(x.shape)
    one_matrix = torch.ones(x.shape)
    x = torch.where(x > 0.5, one_matrix, zero_matrix)
    y = torch.where(y > 0.5, one_matrix, zero_matrix)
    count_x = torch.sum(x, dim=1)
    count_y = torch.sum(y, dim=1)
    x_union_y = torch.sum(torch.where(x + y > 0, one_matrix, zero_matrix), dim=1)
    x_intersection_y = (count_x + count_y - x_union_y) / 2
    acc = 2 * torch.div(x_intersection_y, count_x + count_y)
    zero_acc = torch.zeros(acc.shape)
    acc = torch.where(torch.isnan(acc), zero_acc, acc)
    return torch.mean(acc)


def train_model(model, optimizer, data_idx, loss_fn, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0

    # model.cuda()
    model.train()
    transform = transforms.Compose([PrepareData()])
    data_set = SIIMDataset(fold_id=data_idx, transform=transform)
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True)
    for idx, batch in enumerate(data_loader):
        inputs = batch["input"]
        target = batch["target"]
        # target = target.unsqueeze(0)
        target = torch.autograd.Variable(target)
        # if torch.cuda.is_available():
            # inputs = inputs.cuda()
            # target = target.cuda()
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = loss_fn(prediction, target)
        acc = calculate_accuracy(x=prediction, y=target)
        loss.backward()
        optimizer.step()
        steps += 1

        if steps % 1 == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, '
                  f'Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(data_loader), total_epoch_acc / len(data_loader)


def eval_model(model, val_data_idx, loss_fn):
    model.eval()
    transform = transforms.Compose([PrepareData(), HWCtoCHW])
    data_set = SIIMDataset(fold_id=val_data_idx, transform=transform)
    data_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=True)
    with torch.no_grad():
        data = next(data_loader)
        inputs = data["input"]
        target = data["target"]
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()
        prediction = model(inputs)
        loss = loss_fn(prediction, target)
        acc = calculate_accuracy(x=prediction, y=target)

        print(f'Idx: {val_data_idx}', f'Test loss: {loss.item()}', f'Test accuracy: {acc.item()}')

    return loss.item(), acc.item()


if __name__ == '__main__':
    loss = rmse
    for e in range(epochs):
        val_idx = random.randint(0, 9)
        range_list = list(range(0, 10))
        range_list.remove(val_idx)
        for idx in range_list:
            train_model(model=model, optimizer=optimizer, data_idx=idx, loss_fn=loss, epoch=e)
            eval_model(model=model, val_data_idx=idx, loss_fn=loss)
