import numpy as np
import random
import math

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
    print("mean1:",mean_tile_1,"  mean2:",mean_tile_2)
    variance_square_1 = np.sum((train_data_1 - mean_tile_1) * (train_data_1 - mean_tile_1)) / train_data_1.shape[0]
    variance_square_2 = np.sum((train_data_2 - mean_tile_2) * (train_data_2 - mean_tile_2)) / train_data_2.shape[0]
    print("variance_square_1:",variance_square_1,"  variance_square_2:",variance_square_2)
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
    #data_matrix = normalization(data_matrix)
    #data_test = normalization(data_test)
    #data_test = normalization(data_test)
    #data_covariance = np.dot(data_matrix.T,data_matrix)
    #print(data_matrix.shape)
    #eigenvalue,eigenvector = np.linalg.eig(data_covariance)
    #print(eigenvalue)
    #print(eigenvector)
    
    error_sum = naive_bayesian(data_matrix, label_matrix, data_test, label_test)
    print(1 - error_sum)

if __name__ == "__main__":
    main()
