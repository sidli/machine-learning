# The purpose of this program is to assist classmates understand the process of Gradient Descent
# In order to use Gradient Descent, we will need various tools/formulas
#     A Pre-processed Training Set - Price per month of rent at apartment for X months
#    A way to Optimize the Training Set
#     Weights a,b respectively that follow our formula: 1/M SUM ( a*x^(M) + b - y^(m))^2
#        This formula can also be looked at as 1/M SUM (y(prediction) - y)^2
#     A learning step
#     A way to calculate the new a,b with their respective derivatives: 
#        a^(t+1) = a^(t) - learningStep* (derivative(a^(t))
#        b^(t+1) = a^(t) - learningStep * (derivative(b^(t))
#     A way to calculate rent for months rented with the respective weights
#    A way to check if the Sum of Squares Error (SSE) is getting smaller
# The following information comes from the Chatham Courts & Reflections website
# for the volley apartment plan. This isn't exactly a linear graph and the testing sample
# is small, but I hope this program will shed some light on the process anyway.

# The Pre-processed Training Sets
trainingSetX_PreP = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
trainingSetY_PreP = [1044, 1024, 1019, 899, 959, 1019, 944, 904, 899, 969, 909]

# The Processed Training Sets - Haven't been processed yet
trainingSetX_Pro = []
trainingSetY_Pro = []

# The Weights for the Gradient Descent - Random numbers
a = 0.78
b = 0.25

# The Learning Step : Amount to increase / decrease the rates
learningStep = .001

# The previous SSE and current SSE to confirm that SSE is getting smaller
previousSSE = -1
currentSSE = -1

# Function: Calculate the Sum of Squares Error (SSE)
def Calculate_SSE():
    # Temp prev SSE to hold the current SSE before returning
    prevSSE = currentSSE
    sumOfSquares = 0.0

    for i in range(len(trainingSetX_Pro)):
        sumOfSquares += (a * trainingSetX_Pro[i] + b - trainingSetY_Pro[i])**2

    # Temp current SSE to hold the current SSE before returning - Take the Average
    currSSE = (sumOfSquares / len(trainingSetX_Pro))
    return (currSSE, prevSSE)

def Calculate_a():
    sumOfDeriv_a = 0.0
    # The formula for deriv(a ^ (t)) is 1/M SUM 2 * (a * x^(m) + b - y^(m)) * x^(m)
    for i in range(len(trainingSetX_Pro)):
        sumOfDeriv_a += 2 * (a*trainingSetX_Pro[i] + b - trainingSetY_Pro[i]) * trainingSetX_Pro[i]

    averageOfDeriv_a = sumOfDeriv_a / len(trainingSetX_Pro)

    # Following the formula a^(t+1) = a^(t) - learningStep* (derivative(a^(t))
    new_a = a - (learningStep * averageOfDeriv_a)
    return(new_a)

def Calculate_b():
    sumOfDeriv_b = 0.0
    # The formula for deriv(b ^ (t)) is 1/M SUM 2 * (a * x^(m) + b - y^(m))
    for i in range(len(trainingSetX_Pro)):
        sumOfDeriv_b += 2 * (a*trainingSetX_Pro[i] + b - trainingSetY_Pro[i])

    averageOfDeriv_b = sumOfDeriv_b / len(trainingSetX_Pro)

    # Following the formula b^(t+1) = a^(t) - learningStep * (derivative(b^(t)
    new_b = b - (learningStep * averageOfDeriv_b)
    return(new_b)

# We don't want to calculate b using the new a weight, so let's have Calculate_Weights
# Call both and then set the new weights
def Calculate_Weights():
    temp_a = Calculate_a()
    temp_b = Calculate_b()
    return(temp_a, temp_b)

# Lastly, let's check to see if our model can calculate the price based on our weights
def Calculate_Price(months):
    price = a*months + b
    print("Calculating price for ", months, " months. -", price)

# Now that the calculations and variables are defined, we can begin
# Optimize Training Set - X is the same, but we do this for consistency
trainingSetX_Pro = trainingSetX_PreP
for i in range(len(trainingSetY_PreP)):
    trainingSetY_Pro.append(trainingSetY_PreP[i] * trainingSetX_Pro[i])

# With our random Weights, let's calculate the price of the three months
print("With random weights:")
Calculate_Price(3)

# Seems off, so we put everything together to get a closer estimate
currentSSE, previousSSE = Calculate_SSE()

i = 1
while abs(currentSSE - previousSSE) > 0.00001:
    i += 1
    a, b = Calculate_Weights()
    print(a,"  ",b)
    break;
    currentSSE, previousSSE = Calculate_SSE()

print("With our new, better performing weights thanks to Gradient Descent")
# Now that the a,b should lead to closer approximations
for i in range(len(trainingSetX_Pro)):
    Calculate_Price(i+3)

print("a: ", a, " b: ", b)
print("The current Sum Of Squares Error is: ", currentSSE)

