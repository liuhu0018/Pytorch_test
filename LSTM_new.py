import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 16
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
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=dataset_test,
                              batch_size=20,
                              shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=40, num_layers=1, batch_first=True)
        self.conv1d = nn.Conv1d(40, 40, kernel_size=1)
        # self.linear1 = nn.Linear(40, 4)
        # self.relu = nn.ReLU()
        self.linear2 = nn.Linear(40, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = x.squeeze(1)
        # print(x)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # print(x.shape)
        # print(x)
        x = x.squeeze(2)
        # # print(x.shape)
        # x = self.relu(self.linear1(x))
        x = self.linear2(x)
        # print(x.shape)
        return x


model = Model()
writer = SummaryWriter(comment="LSTM_MSE_Adam_LR_0.01_DATASET_15_BATCH_1")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epoch_list = []
loss_list = []


def train():
    for step, (x_train, y_train) in enumerate(train_loader):
        # x_train = x_train.to(device)
        # y_train = y_train.to(device)
        x_train = x_train.unsqueeze(1)
        y_pred = model(x_train)
        # print(y_pred.shape)
        # print('x_train',x_train.shape)
        # print('y_train',y_train.shape)
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
            x_test = x_test.unsqueeze(1)
            y_pred = model(x_test)
            acc = torch.mean(1 - torch.abs(y_pred.data - y_test) / y_test)
        writer.add_scalar(' Acc/test', acc, epoch)
        print("Epoch:{}, Acc:{:.4f}".format(epoch, acc))


if __name__ == '__main__':
    for epoch in range(2000):
        train()
        test()
    # torch.save(model.state_dict(), 'Lstm_params.pth')
