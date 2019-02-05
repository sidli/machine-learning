import random
import numpy as np
import math
import copy

def read_datas(filename):
    datas = []
    labels = []
    for line in open(filename).readlines():
        line_split = list(map(lambda x:float(x), filter(lambda x: x!= "", line.split(','))))
        labels.append(line_split[0])
        datas.append(line_split[1:])
    return np.array(datas), np.array(labels)

def gaussian_prob(datas, mean, covariance):
    datas2 = datas - mean
    p_arr = []
    for item in datas2:
        rlt = 1.0 / math.sqrt(math.pow(2*math.pi, datas.shape[1]) * np.linalg.det(covariance)) * np.exp(-0.5 * np.dot(np.dot(item, np.linalg.inv(covariance)), item.T))
        #rlt = np.linalg.det(covariance) ** -.5 ** (2 * np.pi) ** (datas.shape[1]/2.) * np.exp(-0.5 * np.dot(np.dot(item, np.linalg.inv(covariance)), item.T))
        p_arr.append(rlt)
    return np.array(p_arr)

def GMM(datas,k):
    length,width = datas.shape[0],datas.shape[1]
    #mean_array = [[random.uniform(-1,1) for i in range(width)] for j in range(k)]
    mean_array = datas[np.random.choice(length, k, False), :]
    covariance_matrix_arr = [np.eye(width) for i in range(k)]
    prob = [1.0 / k for i in range(k)]
    q_arr = [[] for i in range(k)]
    log_likelihoods = []
    iteration = 0
    while True:
        iteration += 1
        #E step
        for i in range(k):
            q_arr[i] = prob[i] * gaussian_prob(datas, np.array(mean_array[i]), covariance_matrix_arr[i])
        # Likelihood computation
        log_likelihood = np.sum(np.log(np.sum(q_arr, axis = 0)))
        log_likelihoods.append(log_likelihood)
        q_arr = q_arr / np.sum(np.array(q_arr), axis = 0)
        #M step
        for i in range(k):
            datas2 = []
            for j in range(length):
                datas2.append(datas[j] * q_arr[i,j])
            mean_array[i] = np.sum(np.array(datas2), axis=0) / np.sum(q_arr[i])
            datas3 = []
            for j in range(length):
                datas3.append(np.dot((datas[j] - mean_array[i]).reshape(-1,1), (datas[j] - mean_array[i]).reshape(1,-1)) * q_arr[i,j])
            covariance_matrix_arr[i] = np.sum(np.array(datas3),axis=0) / np.sum(q_arr[i])
            prob[i] = np.sum(q_arr[i]) / length
        #print(log_likelihood)
        if len(log_likelihoods) < 2 : continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < 1: break
        if iteration == 5: break
    #print("iteration:",iteration)
    return log_likelihood

def standardize(datas):
    length = datas.shape[1]
    datas = datas - np.mean(datas, 0)
    variance = 1.0 / length * np. sum(np.square(datas))
    datas = datas / math.sqrt(variance)
    return datas

def main():
    datas,labels = read_datas("leaf.data")
    datas = standardize(datas)
    #print(datas)
    for k in [12,18,24,36,42]:
    #for k in [24]:
        loss_arr = []
        for i in range(20):
            loss_arr.append(GMM(datas, k))
        #print(loss_arr)
        print(k, " mean:", np.mean(np.array(loss_arr)), " variance:", 1.0 / 20.0 * np.sum(np.square(np.array(loss_arr - np.mean(np.array(loss_arr))))))

if __name__ == "__main__":
    main()
