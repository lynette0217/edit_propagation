import cv2
import os
from os import path
import argparse
import numpy as np
import random
import math
from scipy.sparse import dia_matrix


def std_avg(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    std = np.zeros((rows, cols))
    avg = np.zeros((rows, cols))

    matrix_up = matrix[0:1, :]
    matrix_down = matrix[rows-1:, :]
    matrix = np.r_[matrix, matrix_down]
    matrix = np.r_[matrix_up, matrix]

    matrix_left = matrix[:, 0:1]
    matrix_right = matrix[:, rows-1:]
    matrix = np.c_[matrix_left, matrix]
    matrix = np.c_[matrix, matrix_right]

    for i in range(rows):
        for j in range(cols):
            std[i][j] = np.std(matrix[i:i+3, j:j+3])
            avg[i][j] = np.mean(matrix[i:i+3, j:j+3])
            # std[i][j] = 0
            # avg[i][j] = matrix[i][j]
    return std, avg


def basic_calculation(weight_edited, weight_nonedited, image, mask, operation, beta):

    # This function is for the calculation of some basic parameters that the algorithm would use.
    # We calculate g, W and lambda here.

    rows = mask.shape[0]
    cols = mask.shape[1]
    n = rows*cols
    lambda_result = 0  # lambda = \Sigma_i{w_i}/n
    g = np.zeros((n, 3))  # vector g, soft constraints, user-satisfied edits
    W = np.zeros((n, n))  # diagonal matrix W, with W_ii = w_i
    for i in range(rows):
        for j in range(cols):
            if mask[i][j] == 0:
                lambda_result += weight_nonedited
                W[i*cols + j][i*cols + j] = weight_nonedited
                for k in range(3):
                    g[i*cols + j][k] = image[i][j][k]
            elif mask[i][j] == 255:
                lambda_result += weight_edited
                W[i*cols + j][i*cols + j] = weight_edited
                if operation == 0:
                    g[i*cols + j][0] = image[i][j][0] + beta
                    if(g[i*cols + j][0]) > 255:
                        g[i*cols + j][0] = 255
                    g[i*cols + j][2] = image[i][j][2]
                    g[i*cols + j][1] = image[i][j][1]

                elif operation == 1:  # make it blue!
                    g[i*cols + j][2] = image[i][j][2]+beta
                    g[i*cols + j][1] = image[i][j][1]
                    g[i*cols + j][0] = image[i][j][0]
                elif operation == 2:
                    g[i*cols + j][0] = 256
                    g[i*cols + j][1] = 128
                    g[i*cols + j][2] = 128
            else:
                print(
                    "something is wrong about the mask in the lambda_calculate function.")
                exit(0)

    lambda_result /= n
    print("finish the basic calculation!")
    return lambda_result, g, W


def affinity_calculation(image, m, n, sigma_a, sigma_s, img_num):
    U = np.ones((n, m)) * (-1)
    A = np.ones((m, m)) * (-1)
    # Here, we begin to sample matrix U

    # index[i] means that the index[i]th column in Z == the ith column in U
    index = np.ones(m) * (-1)
    if m == n:
        for i in range(m):
            index[i] = i
    elif m != n:
        # index = np.sort(random.sample(range(0, n), m))
        for i in range(m):
            index[i] = int((i*int(n/m)))
            # print(index[i])

    # Now we can use index_list.index(x) to find out that element is index[which],
    # index_list = index.tolist()
    # which means we're able to know column x in Z is which column in U

    # f: pixel color(here we have three channels) + the average and the std of the 3*3 neighbors.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std, avg = std_avg(gray)

    rows = image.shape[1]
    # Note that when we calculate the U[i][j], we actually are calculating the similarity between the index[i]th and index[j]th point.
    for i in range(n):  # row-->doesn't need to sample
        for j in range(m):  # column--->needs to sample
            # print(j)
            if i < j or i == m or i > m or i == j:
                f_i, f_j = np.zeros(5), np.zeros(5)
                f_i[0:3] = image[int(i/rows), int(i % rows), 0:3]
                f_j[0:3] = image[int(index[j]/rows), int(index[j] % rows), 0:3]

                f_i[3], f_i[4] = std[int(i/rows), int(i %
                                                      rows)], avg[int(i/rows), int(i % rows)]
                f_j[3], f_j[4] = std[int(index[j]/rows), int(index[j] %
                                                             rows)], avg[int(index[j]/rows), int(index[j] % rows)]
                left = -(np.square(np.linalg.norm(f_i-f_j)))/sigma_a
                right = - \
                    (np.square(np.linalg.norm(
                        [int(i/rows)-int(index[j]/rows), int(i % rows)-int(index[j] % rows)])))/sigma_s
                U[i][j] = math.exp(left) * math.exp(right)
            elif i > j and i < m:
                U[i][j] = U[j][i]
        if (i % 500 == 0):
            process = int((i/n)*100)
            print("The affinity calculation has been done %d%%" % process)
    print("Finish the affinity matrix calculation!")
    if(np.any(U == -1)):
        print("ERROR:some elements in U matrix didn't get assgined!")
        exit(0)
    if m != n:
        A = U[index.astype('int64'), :]
        if(np.any(A == -1)):
            print("ERROR:some elements in A matrix didn't get assgined!")
            exit(0)
    print("saving affinity matrix")
    file_path = "./matrices/"+"-img=" + \
        str(img_num)+"-m="+str(m)+"-sigma_a=" + \
        str(sigma_a)+"-sigma_s="+str(sigma_s)+"/"
    os.makedirs(file_path)
    np.save(file_path+"U.npy", U)
    np.save(file_path+"A.npy", A)
    return U, A


def appProp_lra_calculation(U, A, g, W, m, n, lambda_result):
    e = np.zeros(n)

    if m == n:
        A = U[:m, :m]

    A_inverse = np.linalg.inv(A)
    U_transpose = U.T
    # Here we calculate matrix D step by step
    # First, we calculate vector d
    ones = np.ones(n)
    D = np.zeros((n, n))
    D_inverse = np.zeros((n, n))
    print("------------------------------------------")
    print("begin the calculation for matrix D")
    if np.all(U == 0) or np.all(A_inverse == 0) or np.all(U_transpose == 0):
        print("ERROR:Here is a zero matrix")
        exit(0)
    right = np.dot(np.dot(U, A_inverse), U_transpose)
    print("finish the first phase of calculating D")
    # left = (1/(2*lambda_result)) * \
    #     np.dot(right, W)
    left = (1/(lambda_result)) * \
        np.dot(right, W)

    print("finish the second phase of calculating D")
    d = np.dot(left+right, ones)
    del left
    for i in range(n):
        D[i, i] = d[i]
    print("finish the D matrix formation")
    print("------------------------------------------")
    D_inverse = np.linalg.pinv(D)
    # Now, we calculate e
    D_inverse_m_u = np.dot(D_inverse, U)
    if m != n:
        matrix_1 = np.linalg.inv((-A + np.dot(U_transpose, D_inverse_m_u)))
        print("the first phase of e vector has finished")
        matrix_2 = np.dot(
            np.dot(np.dot(D_inverse_m_u, matrix_1), U_transpose), D_inverse)
        del matrix_1
        print("the second phase of e vector has finished")
        matrix_3 = np.dot(np.dot(right, W), g)
        print("the third phase of e vector has finished")
        print("------------------------------------------")
        # e = (1/(2*lambda_result))*np.dot((D_inverse - matrix_2), matrix_3)
        e = (1/(lambda_result))*np.dot((D_inverse - matrix_2), matrix_3)

        del matrix_2
        del matrix_3
        print("finish the lra caculation")
        return e
    elif m == n:
        matrix_1 = np.dot(np.linalg.inv(D-U), U)
        matrix_2 = np.dot(matrix_1, W)
        matrix_3 = np.dot(matrix_2, g)
        e = (1/lambda_result) * matrix_3
        del matrix_1
        del matrix_2
        del matrix_3
        return e


def sketch_lra_caculation(U, k, g, W, n, lambda_result):
    return 0
