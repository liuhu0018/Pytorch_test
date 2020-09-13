import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

data = pd.read_excel('C:\\Users\\80232\\Documents\\data\\data1.xlsx')
BATCH_SIZE = 25
x_ = pd.DataFrame(data, columns=['X', 'Y']).values
x = torch.FloatTensor(x_)
# x = torch.autograd.Variable(x, requires_grad=False)
# print(x.shape)
y_ = pd.DataFrame(data, columns=['pwm1', 'pwm2']).values
y = torch.FloatTensor(y_)
# y = torch.autograd.Variable(y, requires_grad=False)
x_train = x[:-2]
y_train = y[:-2]
x_test = x[-6:]
y_test = y[-6:]
print(x_train)
print(y_train)
print(x_test)
print(y_test)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Data.TensorDataset(x_train, y_train)
load = Data.DataLoader(dataset=dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=True)
model = torch.nn.Sequential(
    # torch.nn.BatchNorm1d(2),
    torch.nn.Linear(2, 32),
    # torch.nn.BatchNorm1d(48),
    torch.nn.ReLU(),
    # torch.nn.Linear(128, 64),
    # torch.nn.ReLU(),
    # torch.nn.Linear(64, 32),
    # torch.nn.ReLU(),
    torch.nn.Linear(32, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
    # torch.nn.BatchNorm1d(2)
)
# print(model)
# model.to(device)
writer = SummaryWriter()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
epoch_list = []
loss_list = []

for epoch in range(2000):
    for step, (x_train, y_train) in enumerate(load):
        # x_train = x_train.to(device)
        # y_train = y_train.to(device)
        y_pred = model(x_train)
        loss = loss_func(y_pred, y_train)
        epoch_list.append(epoch)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar(' Loss/train', loss, epoch)
    writer.add_scalar(' Loss/test', loss, epoch)
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data.item()))

# torch.save(model.state_dict(), 'net_params.pkl')

print(model(x_test))
# plt.plot(epoch_list, loss_list)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()
