# -*- coding: utf-8 -*-
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
   
class GMM:
    def __init__(self, k = 3, eps = 0.0001):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`
    
    def fit_EM(self, X, max_iters = 1000):
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape
        
        # randomly choose the starting centroids/means 
        ## as 3 of the points from datasets        
        mu = X[np.random.choice(n, self.k, False), :]
        
        # initialize the covariance matrices for each gaussians
        Sigma= [np.eye(d)] * self.k
        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []
        
        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 
                        
        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            # E - Step
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            print(log_likelihood)
            
            log_likelihoods.append(log_likelihood)
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        return log_likelihood
    
    def predict(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **\
                (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu , \
                        np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in \
            zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)
 
def standardize(datas):
    length = datas.shape[1]
    datas = datas - np.mean(datas, 0)
    variance = 1.0 / length * np. sum(np.square(datas))
    datas = datas / math.sqrt(variance)
    return datas

if __name__ == "__main__":
    datas,labels = read_datas("leaf.data")
    datas = standardize(datas)
    for k in [12,18,24,36,42]:
        loss_arr = []
        for i in range(20):
            gmm = GMM(k, 0.01)
            loss_arr.append(gmm.fit_EM(datas))
        #print(loss_arr)
        print(k, " mean:", np.mean(np.array(loss_arr)), " variance:", 1.0 / 20.0 * np.sum(np.square(np.array(loss_arr  - np.mean(np.array(loss_arr))))))
