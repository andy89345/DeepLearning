import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import time
tf.disable_v2_behavior()
# 用 numpy 亂數產生 100 個點
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# 等等 tensorflow 幫我們慢慢地找出 fitting 的權重值

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0)) #宣告他的shape和他在-1、1之間
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
#loss=tf.keras.losses.SparseCategoricalCrossentropy()
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        plt.plot(x_data, y_data, 'ro', label='Original data')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label='Fitted line')
        plt.legend()
        plt.ion()
        plt.pause(1)
        plt.close()
        
        

# Learns best fit is W: [0.1], b: [0.3]
