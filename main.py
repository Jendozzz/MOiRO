import numpy as np
import matplotlib.pyplot as plt

M1 = np.array([0, 1]).reshape(2,1)
M2 = np.array([-1, -1]).reshape(2,1)
M3 = np.array([1, 0]).reshape(2,1)

s = (2, 200)
n = 200
a, b = -0.5, 0.5

def normalVector():
    S = np.zeros((2,200))
    n = 200
    for i in range(n):
        random_array = np.random.rand(2, 200) - 0.5
        S += random_array
    S /= np.sqrt(n)
    S /= np.sqrt(1/12)
    return S

def generationA(B):
    a00 = np.sqrt(B[0][0])
    a01 = 0
    a10 = B[0][1] / np.sqrt(B[0][0])
    a11 = np.sqrt(B[1][1] - ((B[0][1]) ** 2) / B[0][0])
    A = np.array(([a00, a01], [a10, a11]))
    return A

def generationX(normal, A, M):
    X = None
    for i in range(200):
        tmp = np.dot(A, normal[:, i]).reshape(2, 1) + M
        if X is None:
            X = tmp
        else:
            X = np.concatenate((X, tmp), axis=1)
    return X

def calcB(X, M):
    B = np.array(([0,0], [0,0]), dtype="float64")
    tmp1 = np.array([[0,0]])
    for i in range(200):
        xi = X[0][i]
        yi = X[1][i]
        #xi_T = X[0][i].reshape(1,2)

        tmp1 = np.append(xi, yi)
        tmp1.shape = (2,1)
        tmp = np.dot(tmp1, tmp1.reshape(1,2))
        B += tmp
    B /= 200
    M = M.reshape(2,1)
    M_T = M.reshape(1,2)
    B = B - np.dot(M, M_T)
    return B

def mahalanobis(M0, M1, B):
    M1_M0 = M1 - M0
    M1_M0_T = M1_M0.reshape(1,2)
    B_1 = np.linalg.inv(B)
    result = np.dot(M1_M0_T, B_1)
    result = np.dot(result, M1_M0)
    return result

def bhatachary(M0, M1, B0, B1):
    M1_M0 = M1 - M0
    M1_M0_T = M1_M0.reshape(1, 2)
    B_half_sum = (B1 + B0) / 2
    B_half_sum_1 = np.linalg.inv(B_half_sum)
    det_B_half_sum = np.linalg.det(B_half_sum)
    det_B0 = np.linalg.det(B0)
    det_B1 = np.linalg.det(B1)
    tmp1 = 0.25 * M1_M0_T
    tmp1 = np.dot(tmp1, B_half_sum_1)
    tmp1 = np.dot(tmp1, M1_M0)
    result = tmp1 + 0.5 * np.log(det_B_half_sum/np.sqrt(det_B1 * det_B0))
    return result

def task1():
    normal1 = normalVector()
    B1 = np.array(([0.1, 0], [0, 0.1]))
    A1 = generationA(B1)
    X1 = generationX(normal1, A1, M1)

    newM1 = np.array((np.sum(X1[0, :]), np.sum(X1[1, :]))) / 200
    print("Мат. ожидание 1: ", newM1)
    newB1 = calcB(X1, newM1)
    print("Корреляционная матрица 1: \n", newB1)
    # matplotlib.pyplot.scatter(X1[0,:], X1[1,:])
    # plt.show()

    normal2 = normalVector()
    B2 = np.array(([0.1, 0], [0, 0.1]))
    A2 = generationA(B2)
    X2 = generationX(normal2, A2, M2)
    newM2 = np.array((np.sum(X2[0, :]), np.sum(X2[1, :]))) / 200
    print("Мат. ожидание 2: ", newM2)
    newB2 = calcB(X2, newM2)
    print("Корреляционная матрица 2: \n", newB2)

    print("Расстояние Махаланобиса: ", mahalanobis(newM1, newM2, B1), "\n\n")

    fig, ax = plt.subplots()
    ax.scatter(X1[0, :], X1[1, :], c='b')
    ax.scatter(X2[0, :], X2[1, :], c='r')
    plt.show()
    with open("task1.npy", 'wb') as f:
        np.save(f, [X1, X2])


def task2():
    normal1 = normalVector()
    B1 = np.array(([0.1, 0], [0, 0.01]))
    A1 = generationA(B1)
    X1 = generationX(normal1, A1, M1)
    newM1 = np.array((np.sum(X1[0, :]), np.sum(X1[1, :]))) / 200
    print("Мат. ожидание 1: ", newM1)
    newB1 = calcB(X1, newM1)
    print("Корреляционная матрица 1: \n", newB1)

    normal2 = normalVector()
    B2 = np.array(([0.3, 0], [0, 0.07]))
    A2 = generationA(B2)
    X2 = generationX(normal2, A2, M2)
    newM2 = np.array((np.sum(X2[0, :]), np.sum(X2[1, :]))) / 200
    print("Мат. ожидание 2: ", newM2)
    newB2 = calcB(X2, newM2)
    print("Корреляционная матрица 2: \n", newB2)

    normal3 = normalVector()
    B3 = np.array(([0.09, 0], [0, 0.05]))
    A3 = generationA(B3)
    X3 = generationX(normal3, A3, M3)
    newM3 = np.array((np.sum(X3[0, :]), np.sum(X3[1, :]))) / 200
    print("Мат. ожидание 3: ", newM3)
    newB3 = calcB(X3, newM3)
    print("Корреляционная матрица 3: \n", newB3)

    print("Расстояние Бхатачария M1 и M2: ", bhatachary(newM1, newM2, B1, B2))
    print("Расстояние Бхатачария M2 и M3: ", bhatachary(newM2, newM3, B1, B2))
    print("Расстояние Бхатачария M1 и M3: ", bhatachary(newM1, newM3, B1, B2))

    fig, ax = plt.subplots()
    ax.scatter(X1[0, :], X1[1, :], c='b')
    ax.scatter(X2[0, :], X2[1, :], c='r')
    ax.scatter(X3[0, :], X3[1, :], c='g')
    plt.show()

    with open("task2.npy", 'wb') as f:
        np.save(f, [X1, X2, X3])

task1()
task2()

