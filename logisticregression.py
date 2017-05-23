#encoding=utf8

import tensorflow as tf
import numpy as np
num_epoch = 30 #迭代轮数
#np.linspace（）创建agiel等差数组，元素个素为500
xs=np.linspace(-5,5,500)
ys=np.sin(xs)+np.random.normal(0,0.01,500)


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1],name='weight'))
b = tf.Variable(tf.random_normal([1],name='bias'))

y_ = tf.sigmoid(w*X+b)

loss = -tf.reduce_mean(Y*tf.log(y_))

alpha = 0.01

trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for epoch in range(num_epoch):

    for x,y in zip(xs,ys):
        sess.run(trainer,feed_dict={X:x,Y:y})
    train_cost = sess.run(loss, feed_dict={X: x, Y: y})
    print "train_cost is:", str(train_cost)
    pass