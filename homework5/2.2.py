#!/usr/bin/python
import numpy as np
import math

def loadDataSet(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = list(map(lambda x: float(x),filter(lambda x: x != "",line.strip().split(','))))
        labelArr.append(lineArr[0])
        dataArr.append(lineArr[1:])
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    return dataArr,labelArr

def logistic_regression(train_data,train_label, lamb):
    length,width = train_data.shape[0], train_data.shape[1]
    w = [1.0] * width
    b = 0.0
    step_size = 0.01
    iteration = 0
    while True:
        iteration += 1
        prob_y1 = 1.0 - 1.0 / (1.0 + np.exp(np.sum(train_data * w, axis = 1) + b))
        derivative_b = np.sum((train_label + 1.0) / 2 - prob_y1) - lamb * b
        value_w = np.sqrt(np.mean(np.square(w)))
        derivative_w = np.sum(train_data.T * ((train_label + 1.0) / 2 - prob_y1), axis = 1) - lamb * value_w
        b = b + step_size * derivative_b
        w = w + step_size * derivative_w
        #print("derivative_w:",derivative_w," derivative_b:",derivative_b, " sum:", np.sum(np.abs(derivative_w) + np.abs(derivative_b)))
        #if np.sum(np.abs(derivative_w) + np.abs(derivative_b)) < 0.001:
        if iteration == 1000:
            break
    return w,b

def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0 
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index

def standardize(datas):
    length = datas.shape[1]
    datas = datas - np.mean(datas, 0)
    variance = 1.0 / length * np. sum(np.square(datas))
    datas = datas / math.sqrt(variance)
    return datas

def main():
    train_data,train_label = loadDataSet("park_train.data")
    train_data = standardize(train_data)
    train_label[train_label == 0.0] = -1.0
    test_data,test_label = loadDataSet("park_test.data")
    test_data = standardize(test_data)
    test_label[test_label == 0.0] = -1.0
    for lamb in [0.1]:
        w,b = logistic_regression(train_data, train_label, lamb)
        print(lamb," ",w," ",b)
        predict_label = []
        for item in test_data:
            if np.sum(w * item) + b > 0.0:
                predict_label.append(1)
            else:
                predict_label.append(-1)
        accuracy,indexes = Calculate_accuracy(predict_label, test_label)
        print("accuracy:",accuracy)

if __name__ == "__main__":
    main()

