#!/usr/bin/python
import numpy as np
import heapq 

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

class scaleData:
    def __init__(self,data):
        even_data = np.mean(data,axis=0)
        self.rate_data = 1 / even_data

    def scale(self,data):
        return self.rate_data * data
    
def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index
    
def KNN(line,dataTrain,labelTrain,k,scaler):
    line = scaler.scale(line)
    #print(line)
    distance = np.zeros(len(dataTrain))
    for i in range(len(dataTrain)):
        distance[i] = np.linalg.norm(line - dataTrain[i])
    #distance_k_index = map(distance.index, heapq.nlargest(k, distance))
    distance_k_index = distance.argsort()[0:k]
    labelTrain_k = labelTrain[distance_k_index]
    #here is 0 and 1
    zero_count = len(labelTrain_k[labelTrain_k == 0.0])
    one_count = len(labelTrain_k[labelTrain_k == 1.0])
    if zero_count > one_count: return 0.0
    else: return 1.0
    
def main():
    dataTrain,labelTrain = loadDataSet("park_train.data")
    dataTest,labelTest = loadDataSet("park_validation.data")
    scaler = scaleData(dataTrain)
    dataTrain_opt = scaler.scale(dataTrain)
    #print(dataTrain_opt)
    for k in [1, 5, 11, 15, 21]:
    #for k in [1]:
        labelPredict = []
        for line in dataTest:
            labelPredict.append(KNN(line,dataTrain_opt,labelTrain,k,scaler))
        success_rate,success_index = Calculate_accuracy(labelTest,labelPredict)
        print("success_rate:",success_rate)

if __name__ == "__main__":
    main()
