# coding:utf8


import numpy as np

# y = 2*x1 + 3*x2 + 4
a = 2
b = 3
c = 4

x1 = np.array(range(100))
x2 = np.array(range(20, 120)) + np.random.rand(100)*10%2

y = a*x1 + b*x2 + c

for i in range(len(y)):
    print "%f,%f,%f" % (x1[i], x2[i], y[i])


