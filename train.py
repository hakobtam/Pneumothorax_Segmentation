import torch


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


def train_model(model, optimizer, data, loss_fn, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0

    # model.cuda()
    model.train()

    for idx, batch in enumerate(data):
        inputs = batch["img"]
        inputs = inputs.unsqueeze(1)
        target = batch["mask"]
        target = target.unsqueeze(0)
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

        if steps % 100 == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, '
                  f'Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(data), total_epoch_acc / len(data)


def eval_model(model, val_data, loss_fn):
    model.eval()
    with torch.no_grad():
        inputs = val_data["img"]
        target = val_data["mask"]
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            target = target.cuda()
        prediction = model(inputs)
        loss = loss_fn(prediction, target)
        acc = calculate_accuracy(x=prediction, y=target)

    return loss.item(), acc.item()
