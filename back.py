import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad_(True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print('predict (before training)', 4, forward(4).item)
for epoch in range (100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  #  不会计算计算图  .data  .item()

        w.grad.data.zero_()  # 手工清零
    print('progress:', epoch, l.item())
print('predict (after training)', 4, forward(4).item())

