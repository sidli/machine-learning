import random
import numpy as np
import math

def read_datas(filename):
    datas = []
    labels = []
    for line in open(filename).readlines():
        line_split = list(map(lambda x:float(x), filter(lambda x: x!= "", line.split(','))))
        labels.append(line_split[0])
        datas.append(line_split[1:])
    return np.array(datas), np.array(labels)

def k_means(datas,k):
    length = len(datas)
    k_index = random.sample(range(length), k)
    k_centers = datas[k_index]
    iteration = 0
    while True:
        iteration += 1
        k_cluster = [[] for i in range(k)]
        k_cluster_index = [[] for i in range(k)]
        for i in range(length):
            #print((k_centers - data) * (k_centers - data))
            index = np.argmin((np.sum((k_centers - datas[i]) * (k_centers - datas[i]), axis = 1)))
            k_cluster[index].append(datas[i])
            k_cluster_index[index].append(i)
        k_centers2 = []
        for cluster in k_cluster:
            k_centers2.append(np.mean(np.array(cluster), axis = 0))
        k_centers2 = np.array(k_centers2)
        if(np.sum(np.fabs(k_centers2 - k_centers)) < 0.001):
            break
        else:
            k_centers = k_centers2
    loss = 0.0
    for i in range(k):
        loss += np.sum(np.square(np.array(k_cluster[i]) - k_centers[i]))
    return k_cluster_index

def standardize(datas):
    length = datas.shape[1]
    datas = datas - np.mean(datas, 0)
    variance = 1.0 / length * np.sum(np.square(datas))
    datas = datas / math.sqrt(variance)
    return datas

def main():
    datas,labels = read_datas("leaf.data")
    datas = standardize(datas)
    datas_index = [[] for i in range(36)] 
    print(datas_index)
    for i in range(len(labels)):
        print(int(labels[i]))
        datas_index[int(labels[i]) - 1].append(i)
    print(datas_index)
    for k in [36]:
    #for k in [24]:
        for i in range(20):
            k_cluster_index = k_means(datas, k)
            print(k_cluster_index)
            exit(0)
        #print(loss_arr)
        print(k, " mean:", np.mean(np.array(loss_arr)), " variance:", 1.0 / 20.0 * np.sum(np.square(np.array(loss_arr - np.mean(np.array(loss_arr))))))

if __name__ == "__main__":
    main()
