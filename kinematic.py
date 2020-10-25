import numpy as np

l_0 = 116
l_1 = 80
n = 14
theta_1 = np.arccos(((x**2 + y**2)/(n**2 + l_0**2))**0.5)
beta_1 = np.arcsin(y/(n * l_0 * np.cos(theta_1))) - theta_1
beta_2 = np.pi - 2 * theta_1 - beta_1
