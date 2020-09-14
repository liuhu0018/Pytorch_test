import torch

y_pred = torch.Tensor([[1, 3, 5],
                      [4, 5, 6]])
y_test = torch.Tensor([[1, 3, 5],
                      [4, 5, 6]])

y = torch.mean(1-torch.abs(y_pred.data-y_test)/y_test)

print(y.item())