import cv2
import os
import numpy as np
import random
# image = cv2.imread("./figs/1.png")
# lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# print(lab)
# out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
# cv2.imwrite("./figs/lab.png", out)

n = 1000
m = 20

# Z = np.random.randint(1, 5, size=(n, n))
# # print(Z)
# columns = np.sort(random.sample(range(0, n), m))
# # columns = np.sort(np.random.randint(0, n, m))  # U's columns
# # print(Z)
# U = Z[:, columns]
# A = U[columns, :]
# print(U)
# print(A.shape)
# index[i] means that the index[i]th column in Z == the ith column in U
index = np.sort(random.sample(range(0, n), m))
index_list = index.tolist()
print(index_list)
print(index_list.index(index_list[5]))

# U = np.random.randint(1, 5, size=(n, m))
