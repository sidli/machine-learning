import pandas as pd
import numpy as np
from pandas import Series, DataFrame 
import matplotlib.pyplot as plt 
import matplotlib

matplotlib.style.use('ggplot')

X = [1,2,3,4,5,6,7,8,9,10]
Y = [0.7593582887700535,0.7593582887700535,0.7593582887700535,0.7005347593582888,0.7700534759358288,0.7700534759358288,0.7700534759358288,0.7700534759358288,0.7165775401069518,0.7165775401069518]

a_list = []
for i in range(10):
    a_list.append([X[i], Y[i]])

data = pd.DataFrame(a_list, columns = ["iteration","accuracy"])
ax = data.plot.line(x='iteration',y='accuracy')
plt.show()
