import numpy as np
from cvxopt import matrix, solvers
import csv
def loadData():
    data = []
    with open('sonar_train.csv') as fileReader:
        line = fileReader.readline()  # 整行读取数据
        while line:
            p_tmp = [float(i) for i in line.split(",")[0:]]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            data.append(p_tmp)  # 添加新读取的数据
            line = fileReader.readline()  # 整行读取数据
    data = np.array(data)  # 将数据从list类型转换为array类型。
    x = data[:, 0:60]
    x = np.array(x)
    y = data[:, 60]
    for i in range(104):
        if y[i] == 2:
            y[i] = -1
    meandata = np.mean(x, axis=0)  # 计算每一列的平均值
    x = x - meandata  # 均值归一化
    covmat = np.cov(x.transpose())  # 求协方差矩阵
    eigVals, eigVectors = np.linalg.eig(covmat)  # 求解特征值和特征向量
    k = 6
    pca_mat = eigVectors[0:k, 0:]  # 选择第一个特征向量
    pca_mat = np.array(pca_mat)
    x = np.dot(x, pca_mat.T)#降低维度后的x数据
    column = k
    row = 104
    c = 1
    g2 = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            x_i = x[i, j]
            y_i = y[i]
            g2[i][j] = -x_i * y_i
    yy = np.zeros((row, 1))
    for i in range(row):
        yy[i][0] = y[i]
    g = np.zeros((row, row))
    for i in range(row):
        g[i][i] = -1.0
    G1 = np.hstack((g2, -yy, g))
    G2 = np.zeros((row, row+1+column))
    for i in range(row):
        for j in range(row+1+column):
            G2[i][column+1+i] = -1.0
    G = np.vstack((G1, G2))
    q = np.zeros((1, row+1+column))
    for i in range(row+1+column):
        if i > column:
            q[0][i] = c
    h1 = -(np.ones((row, 1)))
    h2 = np.zeros((row, 1))
    h = np.vstack((h1, h2))
    p = np.zeros((row+1+column, row+1+column))
    for i in range(row+1+column):
        if i < column:
            p[i][i] = 1.0
    q = q.T
    p = matrix(p)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    sol = solvers.qp(p, q, G, h)  # 调用优化函数solvers.qp求解
    w = sol['x'][0:column]
    b1 = sol['x'][column+1]
    #error of the error of the learned classifier on the validation set for each k and c pair.
    data3 = []
    with open('sonar_valid.csv') as fileReader:
        line1 = fileReader.readline()  # 整行读取数据
        while line1:
            p_tmp1 = [float(i) for i in line1.split(",")[0:]]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            data3.append(p_tmp1)  # 添加新读取的数据
            line1 = fileReader.readline()  # 整行读取数据
    data3 = np.array(data3)  # 将数据从list类型转换为array类型。
    wn = 0
    hang = 52
    x1 = data3[:, 0:60]
    x1 = np.array(x1)
    y1 = data3[:, 60]
    for i in range(52):
        if y1[i] == 2:
            y1[i] = -1

    meandata = np.mean(x1, axis=0)  # 计算每一列的平均值
    x1 = x1 - meandata  # 均值归一化
    covmat = np.cov(x1.transpose())  # 求协方差矩阵
    eigVals, eigVectors = np.linalg.eig(covmat)  # 求解特征值和特征向量
    pca_mat = eigVectors[0:k, 0:]  # 选择第一个特征向量
    pca_mat = np.array(pca_mat)
    x1 = np.dot(x1, pca_mat.T)  # 降低维度后的x数据
    for i in range(hang):
        yr = np.dot(w.T, x1[i]) + b1
        if abs(yr - y1[i]) > 1:
            wn = wn + 1
    error = wn/hang
    print(error)
loadData();

