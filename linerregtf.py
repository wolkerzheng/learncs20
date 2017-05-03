#encoding=utf8

import tensorflow as tf
"""
线性回归模型：Y  = w*x+b
"""
if __name__ == '__main__':
    num_nodes = 2
    output_nodes = 1
    #None表示任意行数
    X = tf.placeholder(tf.float32,[None,num_nodes],name='X')
    Y = tf.placeholder(tf.float32,[None,output_nodes],name='Y')
    # tf.truncated_normal是正态分布的，主要用于初始化一些W矩阵
    W = tf.Variable(tf.truncated_normal([num_nodes,output_nodes],stddev = 0.1))
    b = tf.Variable(tf.truncated_normal([output_nodes],0.1))

    output = tf.matmul(X,W)+b

    loss = tf.reduce_mean(tf.square(output-Y))

    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    sess = tf.Session()

    init  = tf.global_variables_initializer()

    sess.run(init)

    train_x = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    train_y = [[1.0], [0.0], [0.], [0.]]
    for i in range(1000):
        sess.run([train_step], feed_dict={X: train_x, Y: train_y})

    test_x = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    print(sess.run(output, feed_dict={X: test_x}))