from LSTM_new import model
import torch


def predicate(x):
    model.load_state_dict(torch.load('Lstm_params.pth'))
    pred = model(x)
    return pred


if __name__ == '__main__':
    with torch.no_grad():
        x = torch.Tensor([0.499 ,0.865 ]).view(1, 1, 2)
        print(x.shape)
        result = predicate(x)
        print(result)
