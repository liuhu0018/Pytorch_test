import numpy as np


train_data = np.array([[1, 2],
                      [5, 4],
                      [6, 7],
                      [7, 8]], dtype=np.float32)
np.random.shuffle(train_data)
print(train_data)


print(train_data.dtype)
mean = train_data.mean(axis=0)
print(mean)
train_data -= mean
print(train_data)
std = train_data.std(axis=0)
print(std)
train_data /= std
print(train_data)

# test_data -= mean
# test_data /= std