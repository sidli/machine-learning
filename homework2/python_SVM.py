from sklearn import svm
import numpy as np
data = np.loadtxt("park_train.data", dtype=float, delimiter=',')
y, x = np.split(data, (1,), axis=1)
for c in [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 1e+8]:
    for d in [0.1, 1, 10, 100, 1000]:
        #clf = svm.SVC(C=c, gamma=d, kernel='rbf', verbose=True)
        clf = svm.SVC(C=c, gamma=d, kernel='rbf')
        clf.fit(x, y.ravel())
        #f = open("out.txt", "w") 
        #print(clf.support_vectors_,file = f)
        #print(clf.support_)
        #print(clf.get_params)
        print("Accuracy:",clf.score(x, y.ravel()))
        y_pre = clf.predict(x)

def Calculate_SSE(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        exit(1)
    sumOfSquares = 0.0 
    for i in range(len(y1)):
        sumOfSquares += (y1[i] - y2[i])**2
    return (sumOfSquares / len(y1))

def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index
    
print(Calculate_accuracy(y,y_pre))
#print(clf.decision_function(x))
print(clf.intercept_)
