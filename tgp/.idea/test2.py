import tensorflow as tf
sess1 = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
))
sess2 = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
))

a = tf.Variable(1.)
b = tf.assign(tf.Variable(2.), a+10)
sess1.run(tf.initialize_all_variables())
sess2.run(tf.initialize_all_variables())
sess1.run(tf.assign(a, 100))
print sess1.run(a)
print sess1.run(b)
print sess2.run(a)
print sess2.run(b)
