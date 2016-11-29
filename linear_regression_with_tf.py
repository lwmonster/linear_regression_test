# coding:utf8

#from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/linear_train.dat")
train.columns = ['x1', 'x2', 'y']

n_samples = train.shape[0]
# 学习率
learning_rate = 1
# 迭代次数
training_epochs = 1000
# 每多少次输出一次迭代结果
display_step = 50

X_1 = tf.placeholder(tf.float32)
X_2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w_1 = tf.Variable(np.random.randn(), name="weight1", dtype=tf.float32)
w_2 = tf.Variable(np.random.randn(), name="weight2", dtype=tf.float32)
b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)

pred = tf.add(tf.mul(w_1, X_1), tf.mul(w_2, X_2))
pred = tf.add(pred, b)

# 定义损失函数
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n_samples)
# 使用Adam算法，至于为什么不使用一般的梯度下降算法，一会说
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 初始化所有变量
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for i in range(n_samples):
            (x1, x2, y) = train.iloc[i]
            sess.run(optimizer, feed_dict={X_1:x1, X_2:x2, Y:y})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X_1:x1, X_2:x2, Y:y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(c), "W_1=", sess.run(w_1), "W_2=", sess.run(w_2), "b=", sess.run(b))

    print("Optimization Finished!")






