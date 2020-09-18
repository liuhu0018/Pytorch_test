from MLP_new import model
import torch


def predicate(x):
    model.load_state_dict(torch.load('net_params.pth'))
    pred = model(x)
    return pred


if __name__ == '__main__':
    with torch.no_grad():
        x = torch.Tensor([0.444, 0.971])
        result = predicate(x)
        print(result)
