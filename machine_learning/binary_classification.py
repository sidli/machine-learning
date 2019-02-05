#!/usr/bin/python
#Using Gradient Descent calculate aT and b in loss function for Binary Classification
import random
trainingSetX = []
trainingSetY = []
length = 100

#initialize data for our program
def init_data(): 
    for i in range(length): 
        x1 = random.uniform(0,5) 
        x2 = random.uniform(0,5) 
        trainingSetX.append([x1,x2])                 
        if x1>x2:
            trainingSetY.append(1.0)  
        else: trainingSetY.append(-1.0)

# Function: Calculate the Sum of Squares Error (SSE)
def Calculate_SSE(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0
    for i in range(len(y1)):
        sumOfSquares += (y1[i] - y2[i])**2
    return (sumOfSquares / length)

# Function: Calculate the Linear function parameters a and b (SSE)
# In order to use Binary Classification, we will need various tools/formulas
#    Solve a,b respectively with follow formula for their derivative: aT means vector a's vector transpose
#        perception_loss  = 1/M SUM MAX(0, -y^(m)(aT * x^(m) +b))
#        derivative(a^(t) = 1/M SUM - y^(m) * x^(m)
#        derivative(b^(t) = 1/M SUM - y^(m)
#        a^(t+1) = a^(t) - step_size * (derivative(a^(t))
#        b^(t+1) = a^(t) - step_size * (derivative(b^(t))
def Calculate_a_b(x,y,aT,b,step_size):
    sumOfDeriv_aT = [0.0, 0.0]; 
    sumOfDeriv_b = 0.0
    # The formula for deriv(a ^ (t)) is 1/M SUM 2 * (a * x^(m) + b - y^(m)) * x^(m)
    for i in range(length):
        orig_value = - y[i] * (x[i][0] * aT[0] + x[i][1] * aT[1] + b)
        if orig_value < 0:
            continue
        sumOfDeriv_aT[0] += - y[i] * x[i][0]
        sumOfDeriv_aT[1] += - y[i] * x[i][1]
        sumOfDeriv_b += - y[i]
    print("sumOfDeriv_aT:",sumOfDeriv_aT,"  sumOfDeriv_b:",sumOfDeriv_b)
    aT[0] = aT[0] - step_size * sumOfDeriv_aT[0]
    aT[1] = aT[1] - step_size * sumOfDeriv_aT[1]
    b = b - (step_size * sumOfDeriv_b)
    return aT, b, sumOfDeriv_aT[0] * sumOfDeriv_aT[1], sumOfDeriv_b

def Calculate_correct(aT,b):
    global trainingSetX,trainingSetY,length
    expectingSetY = []
    for i in trainingSetX:
        expectingSetY.append(1 if aT[0]*i[0] + aT[1]*i[1] + b > 0 else -1)
    correct_prediction = 0
    for i in range(length):
        if expectingSetY[i] == trainingSetY[i]:
            correct_prediction += 1
    return correct_prediction,expectingSetY

def main():
    global trainingSetX,trainingSetY,length
    init_data()
    aT = [0.0 ,0.0]; b = 0.0;
    step = 0; step_size = 1.0
    correct_prediction = 0
    expectingSetY = []
    #epsilon = 0.000000001
    print(aT," ",b," ",correct_prediction, " ", step, " ", step_size)
    #there are several methods to terminate the recurrence. At here when the correct prediction stop increase, I think I get the best value
    while True:
        step += 1
        aT,b,grada,gradb = Calculate_a_b(trainingSetX,trainingSetY,aT,b,step_size)
        tmp_cor,temset = Calculate_correct(aT,b)
        print(aT," ",b," ",tmp_cor, " ",grada, " ", gradb, " ", step, " ", step_size)
        if tmp_cor > correct_prediction:
            correct_prediction = tmp_cor
            expectingSetY = temset
        elif abs(grada) == 0.0 and  abs(gradb) == 0.0:
            break
    print(aT," ",b," ",tmp_cor, " ",grada, " ", gradb, " ", step, " ", step_size)
    print("aT:",aT," b:", b)
    print("Training Set X",trainingSetX)
    print("Training Set Y",trainingSetY)
    print("Expecting Set Y",expectingSetY)
    print("Sum of Squares Error:", Calculate_SSE(expectingSetY,trainingSetY))
    exit(0)

if __name__ == "__main__":
    main()
