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
    k_centers_index = random.sample(range(length), k)
    k_centers = datas[k_centers_index]
    iteration = 0
    while True:
        iteration += 1
        k_cluster = [[] for i in range(k)]
        for data in datas:
            #print((k_centers - data) * (k_centers - data))
            index = np.argmin((np.sum((k_centers - data) * (k_centers - data), axis = 1)))
            k_cluster[index].append(data)
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
    return loss
    

def standardize(datas):
    length = datas.shape[1]
    datas = datas - np.mean(datas, 0)
    variance = 1.0 / length * np. sum(np.square(datas))
    datas = datas / math.sqrt(variance)
    return datas

def main():
    datas,labels = read_datas("leaf.data")
    datas = standardize(datas)
    for k in [12,18,24,36,42]:
    #for k in [24]:
        loss_arr = []
        for i in range(20):
            loss_arr.append(k_means(datas, k))
        #print(loss_arr)
        print(k, " mean:", np.mean(np.array(loss_arr)), " variance:", 1.0 / 20.0 * np.sum(np.square(np.array(loss_arr - np.mean(np.array(loss_arr))))))

if __name__ == "__main__":
    main()
