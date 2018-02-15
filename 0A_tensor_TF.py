import tensorflow as tf
import numpy as np

# --------------------------------------------------------
# 1. Tensor
# --------------------------------------------------------
# - Type of tensors -
# The concept of "tensor" in Tensorflow is very confusing for beginners.
# When it says "tf.Tensor", that means "Class Tensor".
# And there are some special type of tensors as follows.

# Special type tensors:
# tf.Variable, tf.constant, tf.placeholder ... (There are more...)

# However, there are regular type of tensors 
# 
# tf.zeros, tf.ones
# Ex: 4 -dimensional tensor
# Dimension index: Batch x height x width x color-channel
my_image = tf.zeros([10, 512, 256, 3])

# These are just regular tensor and you can define special type tensor with regular ones.
tf_var0 = tf.Variable(tf.zeros((2,2)))

# Or, you can use numpy
tf_var1 = tf.Variable(np.zeros((2,2)))

# We have to specify 
# - datatype
# - shape
# To declare a special type tensors. (which is Variable, constant, placeholder... etc)


# - Difference between special tensors -
# (1) tf.Variable 
# - tf.Variable is the only type that can be modified.
# - tf.Variable is designed for weights and bias. Not for feeding data.
# - tf.Variable is stored separately, and may live on a parameter server. 
# - tf.Variable should always be initialized before run.
# - Usually declared by <initial value>, <dtype>, name. (There are more...)
mymat = tf.Variable([[7],[11]], tf.int16, name='cat') 
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)

# (2) tf.constant
# - tf.constant holds values that cannot be changed.
# - tf.constant is also designed for weights and bias, but fixed value. 
# - tf.constant value is stored in the graph and its value is replicated wherever the graph is loaded.
# - tf.constant does not need to be initialized.
const_tensor = tf.constant([[7],[11]], tf.int16, name='cat') 


# - tf.placeholder is designed to store values to be fed, such as images.
# - tf.placeholder will produce an error if evaluated. Its value must be fed using the feed_dict optional argument to Session.run(), Tensor.eval(), or Operation.run().
# - tf.placeholder is usually declared with <dtype>, <data shape>
placeholder_tensor = tf.placeholder(tf.float32, shape=(2, 2))




# --------------------------------------------------------
# Convert Numpy Array to Tensor
# --------------------------------------------------------
numpy_array = np.asarray([[1, 2, 3], [4, 5, 6]], np.float32)
tensorflow_tensor = tf.convert_to_tensor(numpy_array, np.float32)
tf.InteractiveSession()
evaluated_tensor = tensorflow_tensor.eval()
print('Numpy Array: ', numpy_array)
print('Tensorflow Tensor: ', tensorflow_tensor)
print('Evaluated Tensor: ', evaluated_tensor)


# --------------------------------------------------------
# Identifying the Dimension and the Shape 
# --------------------------------------------------------
# The dimension is called "Rank" in Tensorflow.
shape = my_image.shape
print('Dimension(shape) of my_image: ', shape)

# However, after graph run, the variable "rank" will hold the rank value.
rank_var =tf.rank(my_image)
rank_var_evaluated = rank_var.eval()
print("Printing 'rank_var' does not show rank of the variable.")
print('rank_var:', rank_var) 
print('But after graph run (evaluation) it obtains rank value.')
print('Evaluated rank_var :', rank_var_evaluated)

# --------------------------------------------------------
# Shaping Tensor Variable 
# --------------------------------------------------------
tensor0 = tf.ones([2, 3, 4])  # 3D Tensor variable filled with ones.
print('The shape of tensor0: %s' %tensor0.shape)

tensor1 = tf.reshape(tensor0, [2, 12])  # Make "tensor0" matrix into 2 by 12 matrix.
print('The shape of tensor1: %s' %tensor1.shape)

# If you want to let it automatically figure out the rest of the shape, do:
tensor2 = tf.reshape(tensor1, [4, -1])  # Make "tensor1" matrix into 4 by 6 matrix.
print('The shape of tensor2: %s' %tensor2.shape)

tensor3 = tf.reshape(tensor2, [2, 2, -1])  # Make "tensor2" matrix into 4 by 6 matrix.
print('The shape of tensor3: %s' %tensor3.shape)









