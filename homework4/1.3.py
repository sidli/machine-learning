import numpy as np
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
    #datas_zeromean = datas - np.mean(datas, axis = 0)
    #datas_variance = np.sqrt(np.mean(np.square(datas_zeromean),axis = 0))
    #datas_standard = datas_zeromean/datas_variance
    #return datas_standard
    avg = np.mean(datas,axis=0)
    avg = np.tile(avg,(datas.shape[0],1))
    return datas - avg

def train(train_data, train_label, c): 
    solvers.options['show_progress'] = False
    line = len(train_data)
    rx = train_label.reshape(-1,1) * train_data
    P = matrix(np.dot(rx, rx.T).astype(float))
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

def SVM(train_data,train_label,data_test,label_test,c):
        a = np.array(train(train_data,train_label,c))
        #print("a:",a)
        w = np.sum(a * train_label.reshape(-1,1) * train_data, axis = 0)
        #print("w:", w)
        minus = c 
        vector_index = -1
        for i in range(len(a)):
            if(abs(c - 2 * a[i]) < minus):
                vector_index = i 
                minus = abs(c - 2 * a[i])
        b = train_label[vector_index] - np.dot(train_data[vector_index], w)
        #b_arr = train_label - np.dot(train_data, w)
        predict_label = []
        for i in range(len(data_test)):
            predict_label.append(1.0 if np.dot(w.T, data_test[i]) + b > 0.0 else -1.0)
        success_rate,success_index = Calculate_accuracy(label_test, predict_label)
        #print("c:", c , " failed_rate:", 1.0 - success_rate)
        print(1.0 - success_rate)
    
def main():
    data_matrix,label_matrix = read_data("sonar_train.csv")
    #data_orginal = data_matrix
    data_test,label_test = read_data("sonar_test.csv")
    data_matrix = normalization(data_matrix)
    data_test = normalization(data_test)
    #data_test = normalization(data_test)
    data_covariance = np.dot(data_matrix.T,data_matrix)
    #print(data_matrix.shape)
    eigenvalue,eigenvector = np.linalg.eig(data_covariance)
    print(eigenvalue)
    print(eigenvector)
    
    label_matrix[label_matrix == 1.0] = 1.0
    label_matrix[label_matrix == 2.0] = -1.0
    label_test[label_test == 1.0] = 1.0
    label_test[label_test == 2.0] = -1.0
    k = 6
    #print(eigenvalue[0:k+1])
    train_data = np.dot(data_matrix, eigenvector[:,:k])
    test_data = np.dot(data_test, eigenvector[:,:k])
    #train_data = data_covariance * eigenvector[:,k]
    SVM(train_data,label_matrix,test_data,label_test,c=1)
    print("-" * 50)
    SVM(data_matrix,label_matrix,data_test,label_test,c=1)
    SVM(np.dot(data_matrix, eigenvector),label_matrix,np.dot(data_test, eigenvector),label_test,c=1)

if __name__ == "__main__":
    main()
