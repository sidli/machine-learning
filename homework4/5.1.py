import math
import numpy as np
import sklearn.cluster as cluster
import matplotlib.image as mi

def basic_algorithm(datas, variance, k):
    len = datas.shape[0]
    A = np.zeros((len,len))
    para = -0.5 / math.pow(variance,2)
    for i in range(len):
        for j in range(i,len):
            A[i,j] = A[j,i] = math.exp(para * math.pow((datas[i] - datas[j]), 2))
    D = np.zeros((len,len))
    D_tmp = np.sum(A,axis=0)
    for i in range(len):
        D[i,i] = D_tmp[i]
    L = D - A 
    eigenvalue,eigenvector = np.linalg.eig(L)
    V = eigenvector[:,-k:]
    estimator = cluster.KMeans(n_clusters = k)
    C_tmp = estimator.fit_predict(V)
    C = [[] for i in range(k)]
    for i in range(C_tmp.size):
        C[C_tmp[i]].append(i)
    return C

def kmean(datas,k):
    print(datas.shape)
    print(datas)
    estimator = cluster.KMeans(n_clusters = k)
    C_tmp = estimator.fit_predict(datas)
    C = [[] for i in range(k)]
    for i in range(C_tmp.size):
        C[C_tmp[i]].append(i)
    return C

def main():
    train_data = mi.imread("bw.jpg")
    train_data = train_data.reshape(-1,1)
    train_data2 = list(train_data)
    C1 = basic_algorithm(train_data,0.05,2)
    for index in C1[0]:
        train_data2[index] = 255
    for index in C1[1]:
        train_data2[index] = 0
    #imsave(fname, arr)
    #C2 = kmean(train_data,2)
    #for index in C2[0]:
    #    train_data2[index] = 234
    #for index in C2[1]:
    #    train_data2[index] = 0
    train_data2 = np.array(train_data2).reshape(75,100)
    mi.imsave("bw_spectral.jpg", train_data2)

if __name__ == "__main__":
    main()
