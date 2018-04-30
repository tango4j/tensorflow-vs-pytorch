import tensorflow as tf
import numpy as np
import os 
from tensorflow.examples.tutorials.mnist import input_data

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
# batch_size = 2**10

gpu1 = 0
gpu2 = 0

# os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu1)
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu1)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, input_size])
Y = tf.placeholder("float", [None, num_classes])

w_h1 = tf.Variable(tf.random_normal(shape=[input_size, hidden_size], stddev=0.01)) 
w_h2 = tf.Variable(tf.random_normal(shape=[hidden_size, hidden_size], stddev=0.01))
w_o = tf.Variable(tf.random_normal(shape=[hidden_size, num_classes], stddev=0.01))

dropout_input = tf.placeholder('float')
dropout_hidden = tf.placeholder('float')

with tf.device('/device:GPU:' + str(gpu1)):
    
    # for d in ['/device:GPU:0', '/device:GPU:1']:
      # with tf.device(d):
    X = tf.nn.dropout(X, dropout_input)
    h1 = tf.nn.relu(tf.matmul(X, w_h1))

with tf.device('/device:GPU:' + str(gpu2)):

    h1 = tf.nn.dropout(h1, dropout_hidden)
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))

    output = tf.matmul(h2, w_o)
    py_x = tf.nn.softmax(output)


# py_x, h2 = MultilayerNN(X, w_h1, w_h2, w_o, dropout_input, dropout_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
# activation_op = tf.argmax(h2_activation, 1)

# Rather than using --- tf.Session() as sess:
# Only use 30% of available GPU memory
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess_gpu = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess_gpu= tf.Session(config=config)

saver = tf.train.Saver()
# with tf.Session() as sess:
with sess_gpu as sess:
    tf.global_variables_initializer().run()
    for i in range(20):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          dropout_input: 0.8, dropout_hidden: 0.5})
        print('Training Epoch ', i, 'Dev Accuracy: ', np.mean(np.argmax(teY, axis=1) ==  sess.run(predict_op, feed_dict={X: teX, dropout_input: 1.0, dropout_hidden: 1.0})))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)# print('Out output: ', sess.run(h2, feed_dict={X: np.reshape(teX[0], [1, 784]), dropout_input: 1.0, dropout_hidden: 1.0}))

# with sess_gpu2 as sess2:
    # saver.restore(sess2, "./tmp/model.ckpt")
    # # inputX = np.expand_dims(teX[0], axis=0)
    # # activation_out = sess2.run(output, feed_dict={X: inputX, dropout_input: 1.0, dropout_hidden: 1.0})
    # print('Out output: ', sess2.run(h2, feed_dict={X: np.reshape(teX[0], [1, 784]), dropout_input: 1.0, dropout_hidden: 1.0}))


