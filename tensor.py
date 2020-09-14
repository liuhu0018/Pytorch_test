import torch

y_pred = torch.Tensor([[1, 3, 5],
                      [4, 5, 6]])
y_test = torch.Tensor([[1, 3, 5],
                      [4, 5, 8]])

y = torch.mean(1-torch.abs(y_pred.data-y_test)/y_test)
y_1 = torch.mean(y_pred-y_test)

print(y.item(), y_1)