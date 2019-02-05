import numpy as np
import random
'''
def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    fr = open(filename)
    for line in fr.readlines():
        lineArr=list(filter(lambda x: x != "",line.strip().split(',')))
        print(lineArr)
        dataMat.append([])
        for i in range(len(lineArr)-1):
            dataMat[-1].append(float(lineArr[i]))
        labelMat.append(float(lineArr[-1]))
    return dataMat,labelMat
'''
def init_data():
    trainingSetX = [[1.1319336623917824, 1.7458258487347078], [2.693851465636563, 1.6449269341742472], [4.193601065808089, 4.224858413101656], [0.3268057210236869, 1.359166232082597], [1.697169100805012, 3.7093029775318103], [4.720790977318251, 2.9013038006316547], [3.090330953501459, 2.53204323593843], [1.7156489838035194, 0.763638249654125], [4.322801149607832, 0.6147931079787772], [0.6993259504902472, 3.0561024916654613], [0.7964427521250833, 1.312202387357827], [1.9187487747729914, 1.1059816914997782], [0.9307660496126446, 1.3753602009135146], [1.712129613108816, 1.1196679301644985], [0.4851314597440265, 4.392221172994075], [3.35242264192328, 4.842757268989808], [2.6576025116927573, 2.813949329843093], [2.5379065593372014, 2.4226060789504826], [3.181325261597478, 1.699152275691434], [3.3243970388399955, 1.9770992806642225], [4.975392834508917, 3.307694444008627], [2.1012237390137587, 3.8436533227764147], [1.0689048635366132, 3.390445517929093], [0.18764438836798902, 4.568427178488958], [4.505241659834204, 0.4598508829542419], [0.29979256902698814, 2.0664989854002584], [1.8401379223460896, 0.7700327263288126], [3.2578758047532386, 4.4542896391237825], [2.637532400569649, 4.972417345857762], [4.274383985160842, 1.2187234659102497], [4.890689399100834, 2.5956493514886043], [1.7270151524210724, 2.5529246953885654], [3.5441474480540376, 2.069076603639993], [2.9365322974784513, 1.9267702547516918], [1.1721568590186964, 1.2372049954199704], [1.525187133274048, 4.2267539613526175], [3.014042545399176, 1.9012979487592592], [2.693341126309641, 3.820061089967359], [1.1521458593743927, 3.677594345308737], [0.2084085226397886, 1.6404904802865639], [4.244963269931386, 3.3199771104502935], [1.2004126923565317, 3.618038804893126], [1.2113155232672757, 3.031043481849887], [3.8911567656982635, 3.7903877488480013], [4.8808627093987305, 1.9531375238933708], [3.8116312556746896, 2.4486382578225037], [2.182434919097517, 4.292088578656362], [1.7453598846626261, 0.4820797884730338], [1.9191286632191256, 1.6448975760741869], [2.234962925501731, 4.288068877005228], [2.3044624977448276, 3.088516618720796], [3.8958486360358884, 1.1718300393853442], [0.6674801506446032, 2.1293209784290057], [1.9970925140064144, 1.7272188026259965], [4.991850434076821, 3.46479668087555], [4.804083109277399, 4.532575765348609], [2.129119961113331, 4.287938377361727], [2.456377647537792, 4.33784500468423], [4.919427583885083, 1.0935962745341872], [2.330922174815967, 2.151925217382617], [2.6264247491157833, 2.017900124603819], [0.0013657069404621192, 3.446587810372656], [0.23933985262993174, 4.380024870591375], [4.259855683808752, 1.9253417282007663], [4.885756827972573, 4.845850417595], [1.303491967803792, 4.159186143564319], [4.995780013552341, 0.7419994639731564], [1.1849353559395603, 1.7744384763718952], [1.0466362045631254, 4.471066818036594], [3.996228743203814, 0.9028783893930381], [0.1993833942653206, 3.4651587193441613], [4.0442403992962035, 1.3461685159472787], [4.731399707449152, 2.5334113794636903], [1.3139207187368713, 0.8586694885650253], [2.5588639895618535, 3.6280647340131105], [3.7392366564680533, 2.4996792449982994], [4.09197860916286, 4.528538410755437], [2.648426849841339, 1.4137822919198728], [1.780795394216147, 3.3236737516991677], [2.7020395430043047, 1.3665199672394313], [3.6770060268761484, 2.9993438829163215], [1.672186925915129, 3.6896130103887], [4.762576110951407, 4.135346023085462], [1.4352744649757532, 2.839093171086012], [2.9041825286592, 4.028901640305483], [2.308864197736387, 0.8912102155639723], [2.5286797774598853, 0.9002463047318771], [1.1233190713338648, 1.3605738228437532], [4.6187246450754795, 2.5199943035943546], [1.7539732178497702, 1.3138973656029096], [0.9656658198812507, 2.940889970711595], [3.9857242424645967, 4.703451509055173], [3.1877718216388216, 2.2865648271264205], [1.5228571587006874, 2.0172575663238597], [1.980704315865756, 0.1271628506867506], [3.429300883862198, 4.740317149029249], [2.9653741653663035, 1.1440884780908518], [0.16245585621480096, 3.148447878949145], [0.6651539782133548, 2.5447739598522268], [0.14577071532306307, 1.1599910190692824]]
    trainingSetY = [-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    return trainingSetX,trainingSetY

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
        self.Earr = [-labelArr[i] for i in range(self.length)] 

def satisfy_stop(data):
    #condition1: SUM(i from 1 to N) a.i * y.i shuold be 0
    sum_conditions = 0.0
    for i in range(data.length):
        sum_conditions += data.labelArr[i] * data.a[i]
    if sum_conditions > 1e-3:
        print("sum_conditions:",sum_conditions)
        return False
    #condition2: KTT condition
    #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i + b)
    #E.i = g(x.i) - y.i
    gArr = [data.labelArr[i] * (data.Earr[i] + data.labelArr[i]) for i in range(data.length)]
    for i in range(data.length):
        if (gArr[i] == 1.0 and data.a[i] > 0.0) or (gArr[i] > 1.0 and data.a[i] == 0.0):
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
    return (iteration - 1)%100 # temporary realization, above code doesnot work well

def select_a2(data, a1_index):
    E_a2_index = -1
    E_a2 = 0.0
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
        if data.a[j] > 0.0: break;
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
        if data.a[i] != 0.0: sv.append(i)
    print("sv:",sv)
    print("data.a:",data.a)
    print("data.b:",data.b)

def calculateEi(data,i):
    g = 0.0
    #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i) + b
    for j in range(data.length):
        if data.a[j] != 0.0: g += data.a[j] * data.labelArr[j] * array_operate(data.dataArr[j], data.dataArr[i], data.gradation, 1) 
    #E.i = g(x.i) - y.i
    Ei = g + data.b - data.labelArr[i]
    print("g:", g, " data.b:", data.b, " data.labelArr[i]:", data.labelArr[i])
    if abs(Ei) < 1e-3:
        Ei = 0.0
    return Ei

def updateEk(data):
    for i in range(data.length):
        g = 0.0
        #g(x.j) = SUM(j from 1 to N) (a.j * y.j * x.j * x.i) + b
        for j in range(data.length):
            if data.a[j] != 0.0: g += data.a[j] * data.labelArr[j] * array_operate(data.dataArr[j], data.dataArr[i], data.gradation, 1) 
        #E.i = g(x.i) - y.i
        data.Earr[i] = g + data.b - data.labelArr[i]
        if abs(data.Earr[i]) < 1e-3:
            data.Earr[i] = 0.0

def updateA(data, a1_index, a2_index):
    #a2_new = a2_old + y.2(E.1 - E.2) / η
    #η = square(x.1 - x.2)
    eta = array_operate(data.dataArr[a1_index], data.dataArr[a1_index], data.gradation, 1) + array_operate(data.dataArr[a2_index], data.dataArr[a2_index], data.gradation, 1) - 2.0 * array_operate(data.dataArr[a1_index], data.dataArr[a2_index], data.gradation, 1)
    a2_new = data.a[a2_index] + data.labelArr[a2_index] * (data.Earr[a1_index] - data.Earr[a2_index]) / eta
    #print("a2_new:",a2_new,"  data.a[a2_index] - data.a[a1_index]:",data.a[a2_index] - data.a[a1_index])
    if data.labelArr[a1_index] != data.labelArr[a2_index]:
        a2_new = max(0.0, a2_new, data.a[a2_index] - data.a[a1_index])
    elif data.labelArr[a1_index] == data.labelArr[a2_index]:
        a2_new = max(0, a2_new)
        a2_new = min(a2_new, data.a[a2_index] + data.a[a1_index])
        if max(0, a2_new) < min(a2_new, data.a[a2_index] + data.a[a1_index]): a2_new = 0.0
    a1_new = data.a[a1_index] + data.labelArr[a1_index] * data.labelArr[a2_index] * (data.a[a2_index] - a2_new)
    if a1_new <= 1e-3: a1_new =0.0
    if a2_new <= 1e-3: a2_new =0.0
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
        if iteration % 1000 == 0:
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
        if data.a[a1_index] != 0.0: updateB(data, a1_index)
        elif data.a[a2_index] != 0.0: updateB(data, a2_index)
        updateEk(data)
        print_detail_info(data)

def main():
    #dataArr,labelArr = loadDataSet("perceptron.data")
    dataArr,labelArr = init_data()
    #dataArr,labelArr = loadDataSet("train_data.txt")
    data = dataStorage(dataArr, labelArr)
    smo(data)

if __name__ == '__main__':
    main()
