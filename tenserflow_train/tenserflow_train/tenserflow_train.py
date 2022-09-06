import tensorflow as tf
from tensorflow.keras import layers
with tf.name_scope("my"):
    variables = tf.Variable(1)

print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")

variables.assign_add(1)
print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")

variables = tf.Variable(2)
variables.assign_add(1)
print("value tensor: ", variables)
print("value       : ", variables.numpy())
print("------------------------------------------")


a = tf.ones([2, 3,4])
print(a)
print("--------------------------")
a = tf.Variable(a)
a[0, 0,3].assign(10)  #必須先經由tf.Variable宣告後才能這樣賦值
b = a.read_value
print(b)
print("--------------------------")
a = tf.constant(2)
b = tf.constant(3)
print("a + b : ", a.numpy() + b.numpy())
print("Addition with constants: ", a+b)
print("Addition with constants: ", tf.add(a, b))
print("a * b :" , a.numpy() * b.numpy())
print("Multiplication with constants: ", a*b)
print("Multiplication with constants: ", tf.multiply(a, b))
print("----------------------------------------------------")

matrix1 = tf.constant([[3., 4.]])

matrix2 = tf.constant([[2.],[3.]])
print(matrix1)
print("\n")
print(matrix2)
print("\n\n")
# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2) #矩陣乘法--->tf.matmul([[1,2],[3,4]],[[5,6],[7,8]])=[[19,22],[43,50]]
print("Multiplication with matrixes:", product)

# broadcast matrix in Multiplication
product2=tf.multiply(matrix1, matrix2) #非矩陣乘法
print("broadcast matrix in Multiplication:", product2)

print("-------------------------------------------")
#dense--->y=wx+b  (w=權重,b=bias,)
net=tf.keras.layers.Dense(60) #一層連接層裡面有n個神經元
net.build(3) #一行訓練數據有三個特徵
print("net.w : ",format(net.kernel))
print("net.b : ",format(net.bias))