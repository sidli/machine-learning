#!/usr/bin/python
import numpy as np
from cvxopt import matrix, solvers
import math

def loadDataSet(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = list(map(float,filter(lambda x: x != "",line.strip().split(','))))
        labelArr.append(lineArr[0])
        dataArr.append(lineArr[1:])
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    return dataArr,labelArr

def train(K, train_label, c, d):
    solvers.options['show_progress'] = False
    line = len(train_label)
    P = matrix(train_label.reshape(-1,1) * train_label * K)
    q = matrix(-np.ones((line, 1)))
    #G = matrix(-np.eye(line))
    G = matrix(np.append(-np.eye(line), np.eye(line), axis = 0))
    #print("G:",G)
    h = matrix(np.append(np.zeros(line), np.array([c for i in range(line)])))
    #h = matrix(np.zeros(line))
    #print("h:",h)
    A = matrix(train_label.reshape(1, -1))
    b = matrix(np.zeros(1))
    return solvers.qp(P, q, G, h, A, b)['x']

def Gaussian_kernel_matrix(rx1, rx2,d):
    rlt = np.zeros([len(rx1), len(rx2)])
    for i in range(len(rx1)):
        for j in range(len(rx2)):
            rlt[i][j] = np.exp(- np.linalg.norm(rx1[i] - rx2[j])**2/(2 * d ** 2))
    return rlt

def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index

def main():
    toler = 1e-3
    train_data,train_label = loadDataSet("park_train.data")
    train_label[train_label == 0.0] = -1.0
    test_data,test_label = loadDataSet("park_test.data")
    test_label[test_label == 0.0] = -1.0
    for c in [1e+7]:
        for d in [1000]:
    #for c in [1.0]:
        #for d in [5.0]:
            K = Gaussian_kernel_matrix(train_data, train_data, d)
            a = np.array(train(K,train_label,c,d))
            #print("c:",c," d:",d)
            #print("a:",a)
            #vector_index = np.array(np.where(a > toler)[0]).reshape(-1,1)
            #vector_index = np.array(list(set(np.where(a > toler)[0]).intersection(set(np.where(a < c - toler)[0]))))
            #print("vector_index:",vector_index)
            minus = c
            for i in range(len(a)):
                if(abs(c - 2 * a[i]) < minus):
                    vector_index = i
                    minus = abs(c - 2 * a[i])
            b = train_label[vector_index] - np.sum(train_label * a.T * K[vector_index])
            #b_array = train_label[vector_index] - train_label[vector_index] * a[vector_index].T[0] * np.sum(K[vector_index], axis=1)
            #b = np.mean(b_array)
            #b = np.median(b_array)
            #print("b:",b)
            predict_label = []
            for i in range(len(test_data)):
            #for i in range(1):
                #print("Gaussian_kernel_matrix(test_data[i],train_data,d):",a.T * train_label * Gaussian_kernel_matrix(np.array([test_data[i]]),train_data,d))
                predict_label.append(1.0 if np.sum(a.T * train_label * Gaussian_kernel_matrix(np.array([test_data[i]]),train_data,d)[0]) + b > 0.0 else -1.0)
                #y_predict[i] = np.sum(a.T[0] * train_label * K[i])
            success_rate,success_index = Calculate_accuracy(test_label, predict_label)
            print(success_rate)
            #print("success_index:",success_index)

if __name__ == "__main__":
    main()
