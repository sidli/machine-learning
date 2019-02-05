#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on Nov 4, 2010
Update on 2017-05-18
Chapter 5 source file for Machine Learing in Action
@author: Peter/geekidentity/片刻
《机器学习实战》更新地址：https://github.com/apachecn/AiLearning
"""
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    """
    dataMat=[[1.1319336623917824, 1.7458258487347078], [2.693851465636563, 1.6449269341742472], [4.193601065808089, 4.224858413101656], [0.3268057210236869, 1.359166232082597], [1.697169100805012, 3.7093029775318103], [4.720790977318251, 2.9013038006316547], [3.090330953501459, 2.53204323593843], [1.7156489838035194, 0.763638249654125], [4.322801149607832, 0.6147931079787772], [0.6993259504902472, 3.0561024916654613], [0.7964427521250833, 1.312202387357827], [1.9187487747729914, 1.1059816914997782], [0.9307660496126446, 1.3753602009135146], [1.712129613108816, 1.1196679301644985], [0.4851314597440265, 4.392221172994075], [3.35242264192328, 4.842757268989808], [2.6576025116927573, 2.813949329843093], [2.5379065593372014, 2.4226060789504826], [3.181325261597478, 1.699152275691434], [3.3243970388399955, 1.9770992806642225], [4.975392834508917, 3.307694444008627], [2.1012237390137587, 3.8436533227764147], [1.0689048635366132, 3.390445517929093], [0.18764438836798902, 4.568427178488958], [4.505241659834204, 0.4598508829542419], [0.29979256902698814, 2.0664989854002584], [1.8401379223460896, 0.7700327263288126], [3.2578758047532386, 4.4542896391237825], [2.637532400569649, 4.972417345857762], [4.274383985160842, 1.2187234659102497], [4.890689399100834, 2.5956493514886043], [1.7270151524210724, 2.5529246953885654], [3.5441474480540376, 2.069076603639993], [2.9365322974784513, 1.9267702547516918], [1.1721568590186964, 1.2372049954199704], [1.525187133274048, 4.2267539613526175], [3.014042545399176, 1.9012979487592592], [2.693341126309641, 3.820061089967359], [1.1521458593743927, 3.677594345308737], [0.2084085226397886, 1.6404904802865639], [4.244963269931386, 3.3199771104502935], [1.2004126923565317, 3.618038804893126], [1.2113155232672757, 3.031043481849887], [3.8911567656982635, 3.7903877488480013], [4.8808627093987305, 1.9531375238933708], [3.8116312556746896, 2.4486382578225037], [2.182434919097517, 4.292088578656362], [1.7453598846626261, 0.4820797884730338], [1.9191286632191256, 1.6448975760741869], [2.234962925501731, 4.288068877005228], [2.3044624977448276, 3.088516618720796], [3.8958486360358884, 1.1718300393853442], [0.6674801506446032, 2.1293209784290057], [1.9970925140064144, 1.7272188026259965], [4.991850434076821, 3.46479668087555], [4.804083109277399, 4.532575765348609], [2.129119961113331, 4.287938377361727], [2.456377647537792, 4.33784500468423], [4.919427583885083, 1.0935962745341872], [2.330922174815967, 2.151925217382617], [2.6264247491157833, 2.017900124603819], [0.0013657069404621192, 3.446587810372656], [0.23933985262993174, 4.380024870591375], [4.259855683808752, 1.9253417282007663], [4.885756827972573, 4.845850417595], [1.303491967803792, 4.159186143564319], [4.995780013552341, 0.7419994639731564], [1.1849353559395603, 1.7744384763718952], [1.0466362045631254, 4.471066818036594], [3.996228743203814, 0.9028783893930381], [0.1993833942653206, 3.4651587193441613], [4.0442403992962035, 1.3461685159472787], [4.731399707449152, 2.5334113794636903], [1.3139207187368713, 0.8586694885650253], [2.5588639895618535, 3.6280647340131105], [3.7392366564680533, 2.4996792449982994], [4.09197860916286, 4.528538410755437], [2.648426849841339, 1.4137822919198728], [1.780795394216147, 3.3236737516991677], [2.7020395430043047, 1.3665199672394313], [3.6770060268761484, 2.9993438829163215], [1.672186925915129, 3.6896130103887], [4.762576110951407, 4.135346023085462], [1.4352744649757532, 2.839093171086012], [2.9041825286592, 4.028901640305483], [2.308864197736387, 0.8912102155639723], [2.5286797774598853, 0.9002463047318771], [1.1233190713338648, 1.3605738228437532], [4.6187246450754795, 2.5199943035943546], [1.7539732178497702, 1.3138973656029096], [0.9656658198812507, 2.940889970711595], [3.9857242424645967, 4.703451509055173], [3.1877718216388216, 2.2865648271264205], [1.5228571587006874, 2.0172575663238597], [1.980704315865756, 0.1271628506867506], [3.429300883862198, 4.740317149029249], [2.9653741653663035, 1.1440884780908518], [0.16245585621480096, 3.148447878949145], [0.6651539782133548, 2.5447739598522268], [0.14577071532306307, 1.1599910190692824]]
    labelMat=[-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    return dataMat, labelMat


def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
    Args:
        aj  目标值
        H   最大值
        L   最小值
    Returns:
        aj  目标值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """smoSimple

    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        maxIter 退出前最大的循环次数
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    dataMatrix = mat(dataMatIn)
    # 矩阵转置 和 .T 一样的功能
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    # 初始化 b和alphas(alpha有点类似权重值。)
    b = 0
    alphas = mat(zeros((m, 1)))

    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # w = calcWs(alphas, dataMatIn, classLabels)
        # print("w:", w)

        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # print('alphas=', alphas)
            # print('labelMat=', labelMat)
            # print('multiply(alphas, labelMat)=', multiply(alphas, labelMat))
            # 我们预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(labelMat[i])
            print("a1 index:",i," fXi:",fXi," Ei:",Ei)

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                #j = selectJrand(i, m)
                j = 1
                # 预测j的结果
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                print("a2 index:",j," fXj:",fXj," Ei:",Ej)
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没发优化了
                if L == H:
                    print("L==H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                print("eta:",eta)
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                print("a1_new:",alphas[i],"a2_new:",alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                print("b_new:",b)
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集

    Returns:
        wc  回归系数
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotfig_SVM(xMat, yMat, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = mat(xMat)
    yMat = mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


if __name__ == "__main__":
    # 获取特征和目标变量
    dataArr, labelArr = loadDataSet('../../../input/6.SVM/testSet.txt')
    # print(labelArr)

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print('/n/n/n')
    print('b=', b)
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # 画图
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)

