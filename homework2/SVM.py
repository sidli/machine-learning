import random

def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    fr = open(filename)
    for line in fr.readlines():
        lineArr=list(filter(lambda x: x != "",line.strip().split(',')))
        dataMat.append([])
        labelMat.append(float(lineArr[0]))
        for i in range(len(lineArr)-1):
            dataMat[-1].append(float(lineArr[i+1]))
    return dataMat,labelMat

class dataStorage:
    def __init__(self, dataArr, labelArr):
        if len(dataArr) != len(labelArr) or len(dataArr) == 0:
            return 1
        self.dataArr = dataArr
        self.labelArr = labelArr
        self.length = len(dataArr)
        self.gradation = len(dataArr[0])
        self.a = [0.0 for i in range(self.length)]
        self.b = 0.0
        self.C = 1.0
        self.toler = 1e-3
        self.Earr = [-labelArr[i] for i in range(self.length)] 

def satisfy_stop(data):
    #condition1: SUM(i from 1 to N) a.i * y.i shuold be 0
    sum_conditions = 0.0
    for i in range(data.length):
        sum_conditions += data.labelArr[i] * data.a[i]
    if sum_conditions > data.toler:
        print("sum_conditions:",sum_conditions)
        return False
    #condition2: KTT condition
    #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i + b)
    #E.i = g(x.i) - y.i
    gArr = [data.labelArr[i] * (data.Earr[i] + data.labelArr[i]) for i in range(data.length)]
    for i in range(data.length):
        if (gArr[i] == 1.0 and data.a[i] > data.toler) or (gArr[i] > 1.0 and data.a[i] == 0.0):
            continue
        else:
            print("Failed iter:",i) 
            return False
    return True

def select_a1(data,iteration):
    '''
    gArr = [data.labelArr[i] * (data.Earr[i] + data.labelArr[i]) for i in range(data.length)]
    for i in range(data.length):
        if data.a[i] > 0.0 and gArr[i] != 1.0:
            return i
    for i in range(data.length):
        if (gArr[i] > 1.0 and data.a[i] == 0.0) or (gArr[i] == 1.0 and data.a[i] > 0.0):
            continue
        else: 
            print("gArr:",gArr[i],"  data.a:",data.a[i])
            return i
    return -1
    '''
    return (iteration - 1) % data.length # temporary realization, above code doesnot work well

def select_a2(data, a1_index):
    E_a2_index = -1
    E_a2 = 0.0
    for i in range(data.length):
        if abs(data.Earr[i]) > data.toler and abs(data.a[i]) > data.toler:
            E_a2_index = i
    if E_a2_index != -1:
        print("select a2_index from KKT vialation")
        return E_a2_index
    for i in range(data.length):
        if data.Earr[i] * data.Earr[a1_index] < 0.0 and abs(data.Earr[i] - data.Earr[a1_index]) > E_a2:
            E_a2 = abs(data.Earr[i] - data.Earr[a1_index])
            E_a2_index = i
    if E_a2_index != -1: 
        print("select a2_index from biggest distance")
        return E_a2_index
    E_a2_index = a1_index
    while (E_a2_index == a1_index):
        E_a2_index = int(random.randint(0, data.length - 1))
    print("select a2_index from random")
    return E_a2_index

def array_operate(arr1, arr2, grad, ope):
    if ope == 0:
        rlt = []
        for i in range(grad):
            rlt.append(arr1[i] + arr2[i])
        return rlt
    elif ope == 1:
        rlt = 0.0
        for i in range(grad):
            rlt += arr1[i] * arr2[i]
        return rlt
    elif ope == 2:
        rlt = []
        for i in range(grad):
            rlt.append(arr1[i] - arr2[i])
        return rlt
    elif ope == 3:
        rlt = []
        for i in range(grad):
            rlt.append(arr1[i] * arr2)
        return rlt

def Calculate_SSE(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0
    failed_index = [i for i in range(len(y1)) if y1[i] != y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] != y2[i])
    return (sumOfSquares / len(y1)), failed_index

def print_detail_info(data):
    print("Earr:",data.Earr)
    #gArr = [data.labelArr[i] * (data.Earr[i] + data.labelArr[i]) for i in range(data.length)]
    #print("gArr:",gArr)
    # w = sum(from 1 to N) a.i * y.i *x.i
    w = [0.0 for w in range(data.gradation)]
    for i in range(data.length):
        w_i = array_operate(data.dataArr[i], data.labelArr[i] * data.a[i], data.gradation, 3)
        w = array_operate(w, w_i, data.gradation, 0)
    print("w:",w)
    # b = y.j - sum(from 1 to N) y.i * a.i * (x.i * x.j)
    for j in range(data.length):
        if data.a[j] > data.toler: break;
    b = updateB(data, j)
    print("b:",b)
    #print SSE
    y2 = []
    for i in range(data.length):
        y2.append(-1.0 if array_operate(data.dataArr[i], w, data.gradation, 1) + b < 0 else 1.0)
    SSE_rate, SSE_index = Calculate_SSE(y2,data.labelArr)
    print("SSE_rate:", SSE_rate, " SSE_index:", SSE_index)
    for i in SSE_index:
        print("index:", i, " data:", data.dataArr[i], " label:", data.labelArr[i], " Earr[i]:", data.Earr[i], "Ei:", calculateEi(data,i))
        print("a * xi + b:", array_operate(w, data.dataArr[i], data.gradation, 1) + b)
    sv = []
    for i in range(data.length):
        if data.a[i] > data.toler: sv.append(i)
    print("sv:",sv)
    print("data.a:",data.a)
    print("data.b:",data.b)

def calculateEi(data,i):
    g = 0.0
    #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i) + b
    for j in range(data.length):
        if data.a[j] > data.toler: g += data.a[j] * data.labelArr[j] * array_operate(data.dataArr[j], data.dataArr[i], data.gradation, 1) 
    #E.i = g(x.i) - y.i
    Ei = g + data.b - data.labelArr[i]
    print("g:", g, " data.b:", data.b, " data.labelArr[i]:", data.labelArr[i])
    #if abs(Ei) < data.toler:
    #    Ei = 0.0
    return Ei

def updateEk(data):
    for i in range(data.length):
        g = 0.0
        #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i) + b
        for j in range(data.length):
            if data.a[j] > data.toler: g += data.a[j] * data.labelArr[j] * array_operate(data.dataArr[j], data.dataArr[i], data.gradation, 1) 
        #E.i = g(x.i) - y.i
        data.Earr[i] = g + data.b - data.labelArr[i]
        #if abs(data.Earr[i]) < data.toler:
        #    data.Earr[i] = 0.0

def updateA(data, a1_index, a2_index):
    #a2_new = a2_old + y.2(E.1 - E.2) / η
    #η = square(x.1 - x.2)
    eta = array_operate(data.dataArr[a1_index], data.dataArr[a1_index], data.gradation, 1) + array_operate(data.dataArr[a2_index], data.dataArr[a2_index], data.gradation, 1) - 2.0 * array_operate(data.dataArr[a1_index], data.dataArr[a2_index], data.gradation, 1)
    print("eta:",eta)
    a2_new = data.a[a2_index] + data.labelArr[a2_index] * (data.Earr[a1_index] - data.Earr[a2_index]) / eta
    #print("a2_new:",a2_new,"  data.a[a2_index] - data.a[a1_index]:",data.a[a2_index] - data.a[a1_index])
    if (data.labelArr[a1_index] != data.labelArr[a2_index]):
        L = max(0, data.a[a2_index] - data.a[a1_index])
        H = min(data.C, data.C + data.a[a2_index] - data.a[a1_index])
    else:
        L = max(0, data.a[a2_index] + data.a[a1_index] - data.C)
        H = min(data.C, data.a[a2_index] + data.a[a1_index])
    if L >= H: return
    if a2_new > H:
        a2_new = H
    if L > a2_new:
        a2_new = L
    a1_new = data.a[a1_index] + data.labelArr[a1_index] * data.labelArr[a2_index] * (data.a[a2_index] - a2_new)
    #if a1_new <= data.toler: a1_new =0.0
    #if a2_new <= data.toler: a2_new =0.0
    print("a1_new:", a1_new, "  a2_new:", a2_new)
    data.a[a2_index] = a2_new
    data.a[a1_index] = a1_new

def updateB(data, j):
    # b = y.j - sum(from 1 to N) y.i * a.i * (x.i * x.j)
    sum_count = 0.0
    for i in range(data.length):
        sum_count += data.labelArr[i] * data.a[i] * array_operate(data.dataArr[i], data.dataArr[j], data.gradation, 1)
    data.b = data.labelArr[j] - sum_count
    print("b:",data.b)
    return data.b

def smo(data):
    iteration = 0
    print_detail_info(data)
    while not satisfy_stop(data):
        iteration += 1
        print("iteration:",iteration)
        if iteration % 100 == 0:
            break
        print("*"*150)
        #select a1
        a1_index = select_a1(data,iteration)
        #select a2
        a2_index = select_a2(data,a1_index)
        print("a1 index:", a1_index, " data.dataArr:", data.dataArr[a1_index], " data.labelArr:", data.labelArr[a1_index], " data.Earr:", data.Earr[a1_index])
        print("a2 index:", a2_index, " data.dataArr:", data.dataArr[a2_index], " data.labelArr:", data.labelArr[a2_index], " data.Earr:", data.Earr[a2_index])
        if(a1_index == -1 or a2_index == -1 or a1_index == a2_index or data.Earr[a1_index] == data.Earr[a2_index]):
            print("Break in select a1 and a2, a1_index:",a1_index,"  a2_index:",a2_index, "a1_Earr:", data.Earr[a1_index], "  a2_Earr:",data.Earr[a2_index])
            print_detail_info(data)
            continue
        updateA(data,a1_index,a2_index)
        #b.1 = -E.1 - y.1 * K(x.1,x.1) * (a.1.new - a.1.old) - y.2 * K(x.2,x.1) * (a.2.new - a.2.old) + b.old  (same to b2)
        if data.a[a1_index] > data.toler: updateB(data, a1_index)
        elif data.a[a2_index] != data.toler: updateB(data, a2_index)
        updateEk(data)
        print_detail_info(data)

def main():
    dataArr,labelArr = loadDataSet("park_train.data")
    #dataArr,labelArr = loadDataSet("train_data.txt")
    data = dataStorage(dataArr, labelArr)
    smo(data)

if __name__ == '__main__':
    main()
