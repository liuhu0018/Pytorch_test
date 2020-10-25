import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

# BATCH_SIZE = 2
xy = np.loadtxt('100data.csv', delimiter=',', dtype=np.float32)
y_train = torch.from_numpy(xy[:90, :-2])
x_train = torch.from_numpy(xy[:90, -2:])
y_test = torch.from_numpy(xy[-20:, :-2])
x_test = torch.from_numpy(xy[-20:, -2:])
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
dataset_train = Data.TensorDataset(x_train, y_train)
dataset_test = Data.TensorDataset(x_test, y_test)
train_loader = Data.DataLoader(dataset=dataset_train,
                               batch_size=16,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=dataset_test,
                              batch_size=20,
                              shuffle=False)

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epoch_list = []
loss_list = []


def train():
    for step, (x_train, y_train) in enumerate(train_loader):
        # x_train = x_train.to(device)
        # y_train = y_train.to(device)
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        epoch_list.append(epoch)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar(' Loss/train', loss.item(), epoch)
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data.item()))


def test():
    with torch.no_grad():
        for data in test_loader:
            x_test, y_test = data
            y_pred = model(x_test)
            acc = torch.mean(1 - torch.abs(y_pred.data - y_test) / y_test)
        writer.add_scalar(' Acc/test', acc, epoch)
        print("Epoch:{}, Acc:{:.4f}".format(epoch, acc))


if __name__ == '__main__':
    writer = SummaryWriter(comment="128_64_32_8_MSE_Adam_LR_0.01_DATASET_60_BATCH_20")
    # model.load_state_dict(torch.load('MLP_params.pth'))
    for epoch in range(2000):
        train()
        test()
    # torch.save(model.state_dict(), 'MLP_params.pth')
