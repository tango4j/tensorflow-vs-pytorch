import tensorflow as tf
import numpy as np
import os 
from mnist_dataloader import dataLoader
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

def conv2Tensor(arg):
    return tf.convert_to_tensor(arg, dtype=tf.float32)

def WeightInitializer(input_size, hidden_size, num_classes, initdev):
    w1 = initdev * conv2Tensor(np.random.randn(input_size, hidden_size))
    w2 = initdev * conv2Tensor(np.random.randn(hidden_size, hidden_size))
    wo = initdev * conv2Tensor(np.random.randn(hidden_size, num_classes))
    return w1, w2, wo

seed = 0
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Hyper Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 2**10
learning_rate = 0.001
initdev = 0.1

def MultilayerNN(X, Y, w_h1, w_h2, w_o):
    x1 = tf.matmul(X, w_h1)
    h1 = tf.nn.relu(x1)
    x2 = tf.matmul(h1, w_h2)
    h2 = tf.nn.relu(x2)
    py_x = tf.matmul(h2, w_o)
    return py_x 

# Placeholder initialization
X = tf.placeholder("float", [None, input_size])
Y = tf.placeholder("float", [None, num_classes])

# Weight initialization
w1, w2, wo = WeightInitializer(input_size, hidden_size, num_classes, initdev)
w_h1 =  tf.get_variable('w_h1', initializer=w1)
w_h2 =  tf.get_variable('w_h2', initializer=w2)
w_o =  tf.get_variable('w_o', initializer=wo)

# Check one of the weights if it is initialized correctly.
with tf.Session() as sess_check:
    tf.global_variables_initializer().run()    
    print('w_h1:',     sess_check.run(w_h1))

# Get the output from the network
py_x = MultilayerNN(X, Y, w_h1, w_h2, w_o)

# Set loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

# Set optimizer
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cost)

# Set predict operation
predict_op = tf.argmax(py_x, 1)

# GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess_gpu= tf.Session(config=config)

# Weight saver setting
saver = tf.train.Saver()

with sess_gpu as sess:
    tf.set_random_seed(seed)
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        batch_count = 0
        if epoch > 0:
            for (tr_bX, tr_bY) in dataLoader(trX, trY, batch_size):
                # sess.run(train_op, feed_dict={X: tr_bX, Y: tr_bY, dropout_input: 1.0, dropout_hidden: 1.0})
                sess.run(train_op, feed_dict={X: tr_bX, Y: tr_bY}) 
                batch_count += 1
        print('Tensorflow: Training Epoch ', epoch+1, 'Test Accuracy: ', np.mean(np.argmax(teY, axis=1) ==  sess.run(predict_op, feed_dict={X: teX})))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)  

