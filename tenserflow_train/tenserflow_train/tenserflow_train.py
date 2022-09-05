import tensorflow as tf
with tf.name_scope("my"):
    variables = tf.Variable(1)

print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")
#  #===========输出: ====================================
#  value tensor:  <tf.Variable 'my/Variable:0' shape=() dtype=int32, numpy=1>
#  value       :  1
#  #=====================================================
# 要在TensorFlow图中使用tf.Variable的值，只需将其视为普通tf.Tensor
variables.assign_add(1)
print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")
#  #===========输出: ====================================
#  value tensor:  tf.Tensor(2, shape=(), dtype=int32)
#  value       :  2
#  #=====================================================
# 要为变量赋值，请使用方法assign，assign_add
variables = tf.Variable(2)
variables.assign_add(1)
print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")
#  #===========输出: ====================================
#  value tensor:  <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
#  value       :  3
#  #=====================================================

a = tf.ones([2, 3,4])
print(a)
print("--------------------------")
a = tf.Variable(a)
a[0, 0,3].assign(10)
b = a.read_value
print(b)
print("--------------------------")