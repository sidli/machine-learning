#!/usr/bin/python
import numpy as np
from cvxopt import matrix, solvers

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
    print("dataArr:",dataArr)
    print("labelArr:",labelArr)
    return dataArr,labelArr

def train(train_data, train_label, c):
    solvers.options['show_progress'] = False
    line = len(train_data)
    rx = train_label.reshape(-1,1) * train_data
    P = matrix(np.dot(rx, rx.T))
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
    validate_data,validate_label = loadDataSet("park_validation.data")
    validate_label[validate_label == 0.0] = -1.0
    for c in [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 1e+8]:
    #for c in [1e+2, 1e+3]:
        a = np.array(train(train_data,train_label,c))
        #print("a:",a)
        w = np.sum(a * train_label.reshape(-1,1) * train_data, axis = 0)
        #print("w:", w)
        vector_index = np.array(np.where(a > toler)[0]).reshape(-1,1)
        b_array = train_label[vector_index] - np.dot(train_data[vector_index], w)
        #b = np.mean(b_array)
        b = np.median(b_array)
        #vector_index2 = np.where(a == np.max(a))[0]
        #print("vector_index2",vector_index2)
        #b2 = train_label[vector_index2] - np.dot(train_data[vector_index2], w)
        #print("b2:",b2)
        predict_label = []
        for i in range(len(validate_data)):
            predict_label.append(1.0 if np.dot(w.T, validate_data[i]) + b > 0.0 else -1.0)
        success_rate,success_index = Calculate_accuracy(validate_label, predict_label)
        print("success_rate:",success_rate)
        #print("success_index:",success_index)

if __name__ == "__main__":
    main()
