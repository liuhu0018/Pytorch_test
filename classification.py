import torchvision

train_set = torchvision.datasets.MNIST(root='../dataset/minist', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='../dataset/minist', train=False, download=True)
