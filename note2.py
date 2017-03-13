#encoding=utf8
import tensorflow as tf

# a = tf.constant(2)
# b = tf.constant(3)
# x = tf.add(a,b)
# with tf.Session() as sess:
#     print sess.run(x)


# a = tf.constant(2)
# b = tf.constant(3)
# x = tf.add(a,b)
# with tf.Session() as sess:
#
#     writer = tf.summary.FileWriter('./graphs',sess.graph)
#
#     print sess.run(x)
#
# writer.close()

# a = tf.constant([2,2],name='a')
# b = tf.constant([[0,1],[2,3]],name='b')
# x = tf.add(a, b,name='add')
# y = tf.mul(a,b,name = 'mul')
#
# with tf.Session() as sess:
#     # writer = tf.summary.FileWriter('./graphs', sess.graph)
#
#     x,y= sess.run([x,y])
#     print x,y

# t_1 = ['apple', 'peach', 'banana']
# a= tf.zeros_like(t_1)
# print a
# print tf.range(10,13)

# my_const = tf.constant([1.0, 2.0], name="my_const")
# with tf.Session() as sess:
#     print sess.graph.as_graph_def()
#
#
# a = tf.Variable(2, name="scalar")
# b = tf.Variable([2, 3], name="vector")
# print a,b


# W = tf.Variable(10)
# W.assign(100)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print W.eval() #10

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
for _ in range(10):
    sess.run(tf.add(x, y)) # create the op add only when you need to compute it
