#!/usr/bin/python

# The purpose of this program is try to solve Binary Classification for ...
# In order to use Gradient Descent, we will need various tools/formulas
#    Solve a,b respectively with follow formula for their derivative: 
#        derivative(a^(t) = 1/M SUM 2 * (a * x^(m) + b - y^(m)) * x^(m)
#        derivative(b^(t) = 1/M SUM 2 * (a * x^(m) + b - y^(m))
#        a^(t+1) = a^(t) - step_size * (derivative(a^(t))
#        b^(t+1) = a^(t) - step_size * (derivative(b^(t))
# A way to check if the Sum of Squares Error (SSE) is getting smaller
# The following information comes from the Chatham Courts & Reflections website
# for the volley apartment plan. This isn't exactly a linear graph and the testing sample
# is small, but I hope this program will shed some light on the process anyway.
# Refer Omeed's Gradient Descent program

# The Pre-processed Training Sets

trainingSetX = []
trainingSetY = []
length = 100

def init_data():
    for i in range(length):
        x1 = random.uniform(0,10)
        x2 = random.uniform(0,10)
        trainingSetX.append([x1,x2])
        if x1>x2:
            trainingSetY.append(1)
        else trainingSetY.append(0)

# Function: Calculate the Sum of Squares Error (SSE)
def Calculate_SSE(y1, y2):
    sumOfSquares = 0.0
    for i in range(length):
        sumOfSquares += (y1[i] - y2[i])**2
    return (sumOfSquares / length)

# Function: Calculate the Linear function parameters a and b (SSE)
def Calculate_a_b(x,y,a,b,step_size):
    sumOfDeriv_a = 0.0
    sumOfDeriv_b = 0.0
    # The formula for deriv(a ^ (t)) is 1/M SUM 2 * (a * x^(m) + b - y^(m)) * x^(m)
    for i in range(length):
        tmp = 2 * (a*x[i] + b - y[i])
        sumOfDeriv_a += tmp * x[i]
        sumOfDeriv_b += tmp
    averageOfDeriv_a = sumOfDeriv_a / length
    averageOfDeriv_b = sumOfDeriv_b / length
    # Following the formula a^(t+1) = a^(t) - learningStep* (derivative(a^(t))
    return a - (step_size * averageOfDeriv_a),b - (step_size * averageOfDeriv_b), averageOfDeriv_a, averageOfDeriv_b

def main():
    # The Weights for the Gradient Descent Random numbers(random initial numbers), y = a*x + b, grada and gradb for the change of a and b
    init_data()
    a = 1.0; b = 1.0; grada = 1.0; gradb = 1.0
    epsilon = 0.00001
    step = 1; step_size = 0.001
    #while grada**2 + gradb**2 > epsilon:
    print(a," ",b," ",grada, " ", gradb, " ", step, " ", step_size)
    #there are several methods to terminate the recurrence. At here when the tangent line smooth enough, we think we get the x0
    while grada**2 * gradb**2 / (step_size**2) > epsilon:
        step += 1
        a,b,grada,gradb = Calculate_a_b(trainingSetX,trainingSetY,a,b,step_size)
        if step%1000 == 0: 
            print(a," ",b," ",grada, " ", gradb, " ", step, " ", step_size)
    print("a:%f b:%f"%(a,b))
    print("Expect result:", expect_y)
    print("Actual result:", trainingSetY)
    exit(0)

if __name__ == "__main__":
    main()
