import math
import numpy as np
import sklearn.cluster as cluster

def read_data(file):
    datas = []
    labels = []
    for line in open(file).readlines():
        line_arr = list(map(lambda x:float(x), filter(lambda x:x!="", line.split(","))))
        datas.append(line_arr[:-1])
        labels.append(line_arr[-1])
    return np.array(datas),np.array(labels)

def basic_algorithm(datas, variance, k):
    len = datas.shape[0]
    A = np.zeros((len,len))
    for i in range(len):
        for j in range(i,len):
            A[i,j] = A[j,i] = math.exp(-0.5 / math.pow(variance,2) * np.dot(datas[i] - datas[j],datas[i] - datas[j]))
    D = np.zeros((len,len))
    D_tmp = np.sum(A,axis=0)
    for i in range(len):
        D[i,i] = D_tmp[i]
    L = D - A
    print("L:",L)
    eigenvalue,eigenvector = np.linalg.eig(L)
    print("eigenvector:",eigenvector)
    V = eigenvector[:,-k:]
    estimator = cluster.KMeans(n_clusters = k)
    estimator.fit(V)
    C_tmp = estimator.predict(V)
    C = [[] for i in range(k)]
    for i in range(C_tmp.size):
        C[C_tmp[i]].append(i)
    print("C:",C)
    
def main():
    train_data,train_label = read_data("sonar_train.csv")
    basic_algorithm(train_data,1.0,10)
    
if __name__ == "__main__":
    main()
