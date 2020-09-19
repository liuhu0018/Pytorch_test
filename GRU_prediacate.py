from GRU import model
import torch


def predicate(x):
    model.load_state_dict(torch.load('GRU_params.pth'))
    pred = model(x)
    return pred


if __name__ == '__main__':
    with torch.no_grad():
        x = torch.Tensor([0.370 ,0.935 ]).view(1, 1, 2)
        result = predicate(x)
        print(result)
