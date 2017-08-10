#encoding=utf8
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)


#拟合平面
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step,sess.run(W),sess.run(b))

# 打印结果
# 0 [[ 0.04509565  0.42019147]] [ 0.48770535]
# 20 [[ 0.06753213  0.23025203]] [ 0.30213606]
# 40 [[ 0.09304091  0.20712551]] [ 0.30012175]
# 60 [[ 0.09852334  0.20169996]] [ 0.29992741]
# 80 [[ 0.09969102  0.20041133]] [ 0.29995567]
# 100 [[ 0.09993666  0.20010105]] [ 0.29998216]
# 120 [[ 0.09998742  0.20002523]] [ 0.29999375]
# 140 [[ 0.09999764  0.20000641]] [ 0.29999796]
# 160 [[ 0.09999958  0.20000166]] [ 0.29999936]
# 180 [[ 0.09999993  0.20000041]] [ 0.29999983]
# 200 [[ 0.1         0.20000014]] [ 0.29999992]