import numpy as np 
import tensorflow as tf

coeffiecients = np.array([[1], [-20], [25]])

w = tf.Variable([0], dtype=float.32)
x = tf.placeholder(tf.float, [3,1])

W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())

cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

for i in range(10000):
	session.run(train, feed_dict={x:coeffiecients})
print(session.run(w))

X = tf.constant(np.random.randn(3,1), name='X')
W = tf.constant(np.random.randn(4,3), name='W')
b = tf.constant(np.random.randn(4,1), name='b')
Y = tf.add(tf.matmul(W,X), b) #YD: matrix multiplication

w = tf.Variable(value, dtype)
x = tf.placeholder(dtype, size)
cost
train = tf.train
init = tf.global_variables_initializer
session = tf.Session()
session.run(init)
with tf.Session() as session:
	session.run(init)

session.run(algo, feed_dict={x: value})

print(session.run(w))
tf.matmul(w,X)


x = tf.placeholder(tf.float32, name = 'x')
sigmoid = tf.sigmoid(x)
with tf.Session() as sess:
	result = sess.run(sigmoid, feed_dict={x:z})

tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)

sess = tf.Session()
# Run the variables initialization (if needed), run the operations
result = sess.run(..., feed_dict = {...})
sess.close() # Close the session

tf.add(...,...) to do an addition
tf.matmul(...,...) to do a matrix multiplication
tf.nn.relu(...) to apply the ReLU activation

tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})