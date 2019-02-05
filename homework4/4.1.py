import math
import numpy as np
import sklearn.cluster as cluster

def circs():
    X = np.zeros((2, 100))
    y = 0
    i_s = np.arange(0, 2*np.pi, np.pi/25.0)
    for i in i_s:
        X[0, y] = np.cos(i)
        X[1, y] = np.sin(i)
        y += 1
    for i in i_s:
        X[0, y] = 2*np.cos(i)
        X[1, y] = 2*np.sin(i)
        y += 1
    return X

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
    eigenvalue,eigenvector = np.linalg.eig(L)
    V = eigenvector[:,-k:]
    print(V.shape)
    estimator = cluster.KMeans(n_clusters = k)
    C_tmp = estimator.fit_predict(V)
    #C_tmp = estimator.predict(V)
    C = [[] for i in range(k)]
    for i in range(C_tmp.size):
        C[C_tmp[i]].append(i)
    print("C:",C)

def kmean(datas,k):
    estimator = cluster.KMeans(n_clusters = k)
    C_tmp = estimator.fit_predict(datas)
    C = [[] for i in range(k)]
    for i in range(C_tmp.size):
        C[C_tmp[i]].append(i)
    print("C:",C)

def main():
    train_data = circs()
    print(train_data)
    train_data = train_data.T
    datas_zeromean = train_data - np.mean(train_data, axis = 0)
    datas_variance = np.sqrt(np.mean(np.square(datas_zeromean),axis = 0))
    print(datas_variance)
    for variance in [0.1,0.2]:
        basic_algorithm(train_data,variance,2)
    kmean(train_data,2)
    
if __name__ == "__main__":
    main()

