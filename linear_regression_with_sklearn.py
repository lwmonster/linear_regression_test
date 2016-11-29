# coding:utf8

#from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/linear_train.dat")
train.columns = ['x1', 'x2', 'y']

n_samples = train.shape[0]
# 学习率
learning_rate = 0.5
# 迭代次数
training_epochs = 1000
# 每多少次输出一次迭代结果
display_step = 50

train_X = pd.DataFrame()
train_X['x1'] = train['x1']
train_X['x2'] = train['x2']

train_y = train['y']


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression(fit_intercept=True)
lr_clf.fit(train_X, train_y)
print "Linear regression coefficients: ",lr_clf.coef_




