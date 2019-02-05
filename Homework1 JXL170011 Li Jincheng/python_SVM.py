from sklearn import svm
import numpy as np
data = np.loadtxt("mystery.data", dtype=float, delimiter=',')

x, y = np.split(data, (4,), axis=1)
clf = svm.SVC(C=23, kernel='rbf', gamma=20, verbose=True)
clf.fit(x, y.ravel())

f = open("out.txt", "w") 
print(clf.support_vectors_,file = f)
print(clf.get_params)
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

print(Calculate_SSE(y,y_pre))
#print(clf.decision_function(x))
print(clf.intercept_)
