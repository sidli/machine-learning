import numpy as np

def read_data(file_name):
    datas = []
    labels = []
    for line in open(file_name).readlines():
        line_split = list(map(lambda x: float(x),filter(lambda x: x!="", line.split(","))))
        datas.append(line_split[:-1])
        labels.append(line_split[-1])
    return np.array(datas), np.array(labels)
    
def normalization(datas):
    #datas_nonm = np.zeros(datas.shape)
    #vector = np.mean(datas, axis = 0)
    #for i in range(datas_nonm.shape[0]):
    #    datas_nonm[i] = vector
    #return datas_nonm
    return datas - np.mean(datas, axis = 0)
    
def main():
    data_matrix,label_matrix = read_data("sonar_train.csv")
    data_matrix = normalization(data_matrix)
    data_covariance = np.dot(data_matrix,data_matrix.T)
    #print(data_covariance)
    eigenvalue ,eigenvector = np.linalg.eig(data_covariance)
    print(eigenvalue)
    print(eigenvector)

if __name__ == "__main__":
    main()
