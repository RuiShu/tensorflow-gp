import sys
sys.path.append('/Users/void/Documents/github/tensorflow-gp')
from tgp.np.gaussian_process import GaussianProcess as GP
import numpy as np
import tensorflow as tf

class DoubleAdamOptimizer(tf.train.AdamOptimizer):
  def _valid_dtypes(self):
    return set([tf.float32, tf.float64])

class GaussianProcess(object):
    def __init__(self, ls=.1, amp=.1, noise=1e-2, sess=tf.Session(), dim=1):
      self.sess = sess
      self.ls = tf.Variable(np.array(ls))
      self.amp = tf.Variable(np.array(amp))
      self.noise = tf.Variable(np.array(noise))
      # construct loss computation graph
      self.xp = tf.placeholder(tf.float64, [None, dim])
      self.yp = tf.placeholder(tf.float64, [None, 1])
      self.x_p = tf.placeholder(tf.float64, [None, dim])
      self.L = self.construct_loss_graph()
      # construct prediction computation graph
      self.y_, self.var_ = self.construct_prediction_graph()
      # set up optimizer
      opt = DoubleAdamOptimizer(
        learning_rate=tf.constant(1e-3, tf.float64),
        beta1=tf.constant(0.9, tf.float64),
        beta2=tf.constant(0.999, tf.float64),
        epsilon=tf.constant(1e-8, tf.float64)
      )
      self.opt_op = opt.minimize(self.L, var_list=[self.ls, self.amp])
      self.grad_ls = tf.gradients(self.L, self.ls)
      self.grad_amp = tf.gradients(self.L, self.amp)
      self.init = tf.initialize_all_variables()
      
    def construct_covariance_graph(self, xs, ys=None):
      add_noise = True if ys is None else False
      ys = xs if ys is None else ys
      # Compute covariance matrix K
      xsq = tf.reduce_sum(tf.square(xs), 1)
      ysq = tf.reduce_sum(tf.square(ys), 1)
      xsq = tf.reshape(xsq, tf.pack([tf.shape(xsq)[0], 1]))
      ysq = tf.reshape(ysq, tf.pack([1, tf.shape(ysq)[0]]))
      sqdist = xsq + ysq - 2*tf.matmul(xs, tf.transpose(ys))
      K = tf.square(self.amp) * tf.exp(-0.5*sqdist)
      if add_noise:
        ones = tf.ones(tf.pack([tf.shape(xs)[0]]), dtype=tf.float64)
        K = K + tf.diag(ones)*(1e-9 + tf.square(self.noise))
        # compute loss
        Ki = tf.matrix_inverse(K)
        return K, Ki
      else:
        return K
      
    def construct_loss_graph(self):
      x = self.xp
      y = self.yp
      xs = x/self.ls
      K, Ki = self.construct_covariance_graph(xs)
      yT = tf.transpose(y)
      Kiy = tf.matmul(Ki, y)
      lK = tf.log(tf.matrix_determinant(K))
      L = tf.matmul(yT, Kiy) + lK
      ones = tf.ones(tf.pack([tf.shape(xs)[0]]), dtype=tf.float64)
      L = L/tf.reduce_sum(ones) * 0.5
      return L

    def construct_prediction_graph(self):
      x = self.xp
      y = self.yp
      x_ = self.x_p
      xs = x/self.ls
      xs_ = x_/self.ls
      _, Ki = self.construct_covariance_graph(xs)
      K_ = self.construct_covariance_graph(xs, xs_)
      # compute variance
      K_T = tf.transpose(K_)
      K_Ki = tf.matmul(K_T, Ki)
      var_ = tf.square(self.amp) - tf.reduce_sum(K_T * K_Ki, 1)
      # compute prediction
      y_ = tf.matmul(K_T, tf.matmul(Ki, y))
      return y_, var_
    
    def solve(self, X, y, epochs=20, batch_size=50, train=True):
      self.X = X
      self.y = y
      self.sess.run(self.init)
      if train:
        iterations = epochs * len(X) / batch_size
        epochiter = iterations/epochs
        for i in xrange(iterations):
          idx = np.random.choice(np.arange(len(X)), batch_size, replace=False)
          X_mini = X[idx]
          y_mini = y[idx]
          fd = {self.xp: X_mini, self.yp: y_mini}
          if i % epochiter == 0:
            print self.sess.run(self.L, feed_dict=fd)
          self.sess.run(self.opt_op, fd)

    def predict(self, X_):
      fd = {self.xp: self.X, self.yp: self.y, self.x_p: X_}
      y_ = self.sess.run(self.y_, feed_dict=fd)
      var_ = self.sess.run(self.var_, feed_dict=fd)
      return y_, var_
        
def main():
  import matplotlib.pyplot as plt
  dim = 1
  X = np.random.uniform(0, 1, (10, dim))
  y = (np.sin((X[:, 0])*10) - X[:,0]*3).reshape(-1, 1)

  ls = [.1]*dim
  sess = tf.Session()
  gp = GaussianProcess(dim=dim, sess=sess, ls=ls, noise=0.1)
  gp.solve(X, y, epochs=100, batch_size=len(X), train=False)
  print sess.run(gp.ls)
  
  X_ = np.linspace(0, 1, 1000).reshape(-1, 1)
  y_, var_ = gp.predict(X_)
  std_ = np.sqrt(var_).reshape(-1, 1)

  plt.scatter(X, y)
  plt.plot(X_, y_, c='r')
  plt.plot(X_, y_+2*std_, c='g')
  plt.plot(X_, y_-2*std_, c='g')
  plt.show()
  

if __name__ == "__main__":
  main()
