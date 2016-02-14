import tensorflow as tf

sess1 = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
))
sess2 = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
))

a = tf.Variable(1.)
d = tf.Variable([1., 1.])
add = tf.assign_add(a, 1.)
b = tf.assign(tf.Variable(a, trainable=True), a)
c = tf.convert_to_tensor(a)
sess1.run(tf.initialize_all_variables())
sess2.run(tf.initialize_all_variables())
print sess1.run(a)
print sess1.run(b)
sess1.run(add)
sess1.run(add)
sess1.run(add)
print sess1.run(b)
print sess1.run(c)
print tf.trainable_variables()

loss_a = tf.square(a)
loss_b = (b - d[0])*(10 - d[0])
loss_c = tf.square(c)

print sess1.run(loss_b)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss_b)

for _ in xrange(100):
    sess1.run(train)


print sess1.run(a)
print sess1.run(b)
print sess1.run(c)
print sess1.run(loss_b)
print sess1.run(d)
print sess2.run(b)
