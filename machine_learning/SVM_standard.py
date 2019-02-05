import sys
from numpy import * 

def loadDataSet(filename): #读取数据
    dataMat=[[1.1319336623917824, 1.7458258487347078], [2.693851465636563, 1.6449269341742472], [4.193601065808089, 4.224858413101656], [0.3268057210236869, 1.359166232082597], [1.697169100805012, 3.7093029775318103], [4.720790977318251, 2.9013038006316547], [3.090330953501459, 2.53204323593843], [1.7156489838035194, 0.763638249654125], [4.322801149607832, 0.6147931079787772], [0.6993259504902472, 3.0561024916654613], [0.7964427521250833, 1.312202387357827], [1.9187487747729914, 1.1059816914997782], [0.9307660496126446, 1.3753602009135146], [1.712129613108816, 1.1196679301644985], [0.4851314597440265, 4.392221172994075], [3.35242264192328, 4.842757268989808], [2.6576025116927573, 2.813949329843093], [2.5379065593372014, 2.4226060789504826], [3.181325261597478, 1.699152275691434], [3.3243970388399955, 1.9770992806642225], [4.975392834508917, 3.307694444008627], [2.1012237390137587, 3.8436533227764147], [1.0689048635366132, 3.390445517929093], [0.18764438836798902, 4.568427178488958], [4.505241659834204, 0.4598508829542419], [0.29979256902698814, 2.0664989854002584], [1.8401379223460896, 0.7700327263288126], [3.2578758047532386, 4.4542896391237825], [2.637532400569649, 4.972417345857762], [4.274383985160842, 1.2187234659102497], [4.890689399100834, 2.5956493514886043], [1.7270151524210724, 2.5529246953885654], [3.5441474480540376, 2.069076603639993], [2.9365322974784513, 1.9267702547516918], [1.1721568590186964, 1.2372049954199704], [1.525187133274048, 4.2267539613526175], [3.014042545399176, 1.9012979487592592], [2.693341126309641, 3.820061089967359], [1.1521458593743927, 3.677594345308737], [0.2084085226397886, 1.6404904802865639], [4.244963269931386, 3.3199771104502935], [1.2004126923565317, 3.618038804893126], [1.2113155232672757, 3.031043481849887], [3.8911567656982635, 3.7903877488480013], [4.8808627093987305, 1.9531375238933708], [3.8116312556746896, 2.4486382578225037], [2.182434919097517, 4.292088578656362], [1.7453598846626261, 0.4820797884730338], [1.9191286632191256, 1.6448975760741869], [2.234962925501731, 4.288068877005228], [2.3044624977448276, 3.088516618720796], [3.8958486360358884, 1.1718300393853442], [0.6674801506446032, 2.1293209784290057], [1.9970925140064144, 1.7272188026259965], [4.991850434076821, 3.46479668087555], [4.804083109277399, 4.532575765348609], [2.129119961113331, 4.287938377361727], [2.456377647537792, 4.33784500468423], [4.919427583885083, 1.0935962745341872], [2.330922174815967, 2.151925217382617], [2.6264247491157833, 2.017900124603819], [0.0013657069404621192, 3.446587810372656], [0.23933985262993174, 4.380024870591375], [4.259855683808752, 1.9253417282007663], [4.885756827972573, 4.845850417595], [1.303491967803792, 4.159186143564319], [4.995780013552341, 0.7419994639731564], [1.1849353559395603, 1.7744384763718952], [1.0466362045631254, 4.471066818036594], [3.996228743203814, 0.9028783893930381], [0.1993833942653206, 3.4651587193441613], [4.0442403992962035, 1.3461685159472787], [4.731399707449152, 2.5334113794636903], [1.3139207187368713, 0.8586694885650253], [2.5588639895618535, 3.6280647340131105], [3.7392366564680533, 2.4996792449982994], [4.09197860916286, 4.528538410755437], [2.648426849841339, 1.4137822919198728], [1.780795394216147, 3.3236737516991677], [2.7020395430043047, 1.3665199672394313], [3.6770060268761484, 2.9993438829163215], [1.672186925915129, 3.6896130103887], [4.762576110951407, 4.135346023085462], [1.4352744649757532, 2.839093171086012], [2.9041825286592, 4.028901640305483], [2.308864197736387, 0.8912102155639723], [2.5286797774598853, 0.9002463047318771], [1.1233190713338648, 1.3605738228437532], [4.6187246450754795, 2.5199943035943546], [1.7539732178497702, 1.3138973656029096], [0.9656658198812507, 2.940889970711595], [3.9857242424645967, 4.703451509055173], [3.1877718216388216, 2.2865648271264205], [1.5228571587006874, 2.0172575663238597], [1.980704315865756, 0.1271628506867506], [3.429300883862198, 4.740317149029249], [2.9653741653663035, 1.1440884780908518], [0.16245585621480096, 3.148447878949145], [0.6651539782133548, 2.5447739598522268], [0.14577071532306307, 1.1599910190692824]]
    labelMat=[-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    return dataMat,labelMat #返回数据特征和数据类别

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): #核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': #线性函数
        K = X * A.T
    elif kTup[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


#定义类，方便存储数据
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征
        self.labelMat = classLabels #数据类别
        self.C = C #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler #停止阀值
        self.m = shape(dataMatIn)[0] #数据行数
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,2))) #缓存
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): #更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值
    print("Evaluation: i",i,"  Ei",Ei)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其E值
        print("Evaluation: j",j,"  Ej",Ej)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109
        updateEk(oS, i) #更新数据
        print("a1:",oS.alphas[j],"a2:",oS.alphas[i])
        print("data.a:",list(oS.alphas))
        print("data.E:",list(oS.eCache))
        #以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        print("b:",oS.b,"  b1_new:",b1,"  b2_new:",b2)
        return 1
    else:
        return 0

#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #遍历所有数据
                alphaPairsChanged = innerL(i,oS)
                #if (i+1) % 21 == 0: sys.exit(0)
                print("fullSet, iter: %d a1_index:%d, a2_index:%d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def testRbf(data_train,data_test):
    dataArr,labelArr = loadDataSet(data_train) #读取训练数据
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3)) #通过SMO算法得到b和alpha
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd] #支持向量的特征数据
    labelSV = labelMat[svInd] #支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0]) #打印出共有多少的支持向量
    m,n = shape(datMat) #训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', 1.3)) #将支持向量转化为核函数
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b  #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        if sign(predict)!=sign(labelArr[i]): #sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m)) #打印出错误率
    dataArr_test,labelArr_test = loadDataSet(data_test) #读取测试数据
    errorCount_test = 0
    datMat_test=mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m,n = shape(datMat_test)
    for i in range(m): #在测试数据上检验错误率
        kernelEval = kernelTrans(sVs,datMat_test[i,:],('rbf', 1.3))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test)/m))

#主程序
def main():
    filename_traindata='train_data.txt'
    filename_testdata='test_data.txt'
    testRbf(filename_traindata,filename_testdata)

if __name__=='__main__':
    main()

