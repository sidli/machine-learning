import numpy as np
#from cvxopt import matrix, solvers

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
    
def main():
    data_matrix,label_matrix = read_data("sonar_train.csv")
    data_matrix_ori = data_matrix
    data_matrix = normalization(data_matrix)
    #print(data_matrix.shape)
    data_covariance = np.dot(data_matrix.T, data_matrix)
    data_covariance = np.dot(data_matrix.T, data_matrix)/(data_matrix.shape[0] - 1)
    #print(data_covariance.shape)
    #print(data_covariance)
    data_covariance2 = np.cov(data_matrix.T)
    #print(data_covariance2.shape)
    #print(data_covariance2)
    eigenvalue,eigenvector = np.linalg.eig(data_covariance)
    print(eigenvalue)
    print(eigenvector)
    
    label_matrix[label_matrix == 1.0] = -1.0
    label_matrix[label_matrix == 2.0] = 1.0
    train_label = label_matrix
    print("------------------------------")
    for k in range(6):
        #print(eigenvalue[0:k+1])
        #train_data = np.dot(data_covariance,eigenvector[0:k+1].T)
        #train_data = data_covariance * eigenvector[:,k]
        #print(eigenvector[:,:k+1])
        train_data = np.dot(data_matrix, eigenvector[:,:k+1])
        for c in [1, 1e+1, 1e+2, 1e+3]:
            #print(train_data)
            #print(train_label)
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
            predict_label = []
            for i in range(len(train_data)):
                predict_label.append(1.0 if np.dot(w.T, train_data[i]) + b > 0.0 else -1.0)
            #print(train_label)
            #print(predict_label)
            success_rate,success_index = Calculate_accuracy(train_label, predict_label)
            #print("k:",k," c:", c , " failed_rate:", 1.0 - success_rate)
            print(1.0 - success_rate)

if __name__ == "__main__":
    main()
