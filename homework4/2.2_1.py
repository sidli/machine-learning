import numpy as np
import random
import math
from cvxopt import matrix, solvers

def read_data(file_name):
    datas = []
    labels = []
    for line in open(file_name).readlines():
        line_split = list(map(lambda x: float(x),filter(lambda x: x!="", line.split(","))))
        datas.append(line_split[:-1])
        labels.append(line_split[-1])
    return np.array(datas), np.array(labels)
    
def normalization(datas):
    avg = np.mean(datas,axis=0)
    avg = np.tile(avg,(datas.shape[0],1))
    return datas - avg

def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0 
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index

def naive_bayesian(train_data,train_label,test_data,test_label):
    train_data_1 = train_data[train_label == 1]
    train_data_2 = train_data[train_label == 2]
    py_1 = len(train_data_1) * 1.0 / len(train_data) 
    py_2 = len(train_data_2) * 1.0 / len(train_data)
    #print(py_1)
    #print(py_2)
    mean_1 = np.mean(train_data_1,axis=0)
    mean_2 = np.mean(train_data_2,axis=0)
    mean_tile_1 = np.tile(mean_1, (train_data_1.shape[0],1))
    mean_tile_2 = np.tile(mean_2, (train_data_2.shape[0],1))
    variance_square_1 = np.sum((train_data_1 - mean_tile_1) * (train_data_1 - mean_tile_1)) / train_data_1.shape[0]
    variance_square_2 = np.sum((train_data_2 - mean_tile_2) * (train_data_2 - mean_tile_2)) / train_data_2.shape[0]
    predict_label = []
    for item in test_data:
        p1 = (1.0 / math.sqrt(2 * math.pi * variance_square_1)) * math.exp( -1.0 / 2.0 * np.dot(item - mean_1, item - mean_1) / variance_square_1)
        p2 = (1.0 / math.sqrt(2 * math.pi * variance_square_2)) * math.exp( -1.0 / 2.0 * np.dot(item - mean_2, item - mean_2) / variance_square_2)
        predict_label.append(1 if p1 * py_1 > p2 * py_2 else 2)
    success_rate,success_index = Calculate_accuracy(predict_label,test_label)
    return (1.0 - success_rate)
    
def main():
    data_matrix, label_matrix = read_data("sonar_train.csv")
    #data_orginal = data_matrix
    data_test, label_test = read_data("sonar_test.csv")
    data_matrix = normalization(data_matrix)
    data_test = normalization(data_test)
    #data_test = normalization(data_test)
    data_covariance = np.dot(data_matrix.T,data_matrix)
    #print(data_matrix.shape)
    eigenvalue,eigenvector = np.linalg.eig(data_covariance)
    print(eigenvalue)
    print(eigenvector)
    
    for k in range(1,11):
        print("k:",k)
        prob_dist = np.mean(np.square(eigenvector[:,:k].T),axis=0)
        error_list = []
        for s in range(1,21):
            error_sum = 0.0
            for i in range(100):
                data_index = []
                for j in range(s):
                    p = random.random()
                    tmp = 0.0
                    index = 0
                    for pi in prob_dist:
                        if tmp + pi > p:
                            break
                        else:
                            tmp = tmp + pi
                            index = index + 1
                    data_index.append(index)
                data_index = list(set(data_index))
                train_data = data_matrix[:,data_index]
                test_data = data_test[:,data_index]
                error_sum += naive_bayesian(train_data,label_matrix,test_data,label_test)
            error_mean = error_sum / 100
            error_list.append(error_mean)
        print(error_list)

if __name__ == "__main__":
    main()
