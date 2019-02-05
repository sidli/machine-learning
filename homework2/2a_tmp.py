#!/usr/bin/python
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def loadDataSet(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = list(map(float,filter(lambda x: x != "",line.strip().split(','))))
        labelArr.append(lineArr[0])
        dataArr.append(lineArr[1:])
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    return dataArr,labelArr

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel=gaussian_kernel, C=None, D=None):
        self.kernel = kernel
        self.C = C
        self.D = D
        if self.C is not None: self.C = float(self.C)
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j], self.D)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        '''
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        '''
        minus = self.C
        for i in range(len(self.a)):
            if(abs(self.C - 2 * self.a[i]) < minus):
                vector_index = i
                minus = abs(self.C - 2 * a[i])
        self.b += self.sv_y[vector_index]
        self.b -= np.sum(self.a * self.sv_y * K[ind[vector_index],sv])
        print("self.b:",self.b)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X, y):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
            #for i in range(1):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv, self.D)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X, y):
        return np.sign(self.project(X,y))

def test_soft():
    X1, y1 = loadDataSet("park_train.data")
    y1[y1 == 0.0] = -1.0
    X_train, y_train = X1,y1
    X_test, y_test = X1,y1
    for c in [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 1e+8]:
        for d in [0.1, 1, 10, 100, 1000]:
            clf = SVM(gaussian_kernel,c,d)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test,y_test)
            correct = np.sum(y_predict == y_test)
            print("%d out of %d predictions correct :" % (correct, len(y_predict)), correct/len(y_predict))

test_soft()
